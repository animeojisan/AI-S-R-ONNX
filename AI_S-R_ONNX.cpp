#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>
#include <deque>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <memory>
#include <limits>
#include <thread>
#include <condition_variable>
#include <functional>
#include <fstream>
#include <sstream>
#include <chrono>
#include <robuffer.h>

#include <d3d11.h>
#include <dxgi1_6.h>
#include <windows.graphics.directx.direct3d11.interop.h>

#include "filter2.h"

#include <wrl/client.h>
#include <winrt/base.h>
#include <winrt/Windows.AI.MachineLearning.h>
#include <winrt/Windows.Foundation.Collections.h>
#include <winrt/Windows.Graphics.h>
#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>
#include <winrt/Windows.Storage.Streams.h>

namespace ml = winrt::Windows::AI::MachineLearning;
namespace wfc = winrt::Windows::Foundation::Collections;
namespace wg = winrt::Windows::Graphics;
namespace wgdx11 = winrt::Windows::Graphics::DirectX::Direct3D11;
namespace wss = winrt::Windows::Storage::Streams;
using Microsoft::WRL::ComPtr;

static uint16_t FloatToHalfBits(float value);
static float HalfBitsToFloat(uint16_t value);


struct __declspec(uuid("5B0D3235-4DBA-4D44-865E-8F1D0E4FD04D")) IMemoryBufferByteAccess : ::IUnknown {
    virtual HRESULT __stdcall GetBuffer(BYTE** value, UINT32* capacity) = 0;
};

static wchar_t g_backend_label_auto[256] = L"GPU優先(自動)";
static wchar_t g_backend_label_gpu0[256] = L"GPU 0";
static wchar_t g_backend_label_gpu1[256] = L"GPU 1";
static wchar_t g_backend_label_cpu[256] = L"CPU固定";

static FILTER_ITEM_SELECT::ITEM g_backend_items[] = {
    { g_backend_label_auto, 0 },
    { g_backend_label_gpu0, 1 },
    { g_backend_label_gpu1, 2 },
    { g_backend_label_cpu, 3 },
    { nullptr }
};
static auto g_backend = FILTER_ITEM_SELECT(L"推論デバイス", 0, g_backend_items);
static auto g_enable = FILTER_ITEM_CHECK(L"有効", true);
static auto g_model_file = FILTER_ITEM_FILE(L"ONNXファイル", L"", L"ONNX Model (*.onnx)\0*.onnx\0All Files (*.*)\0*.*\0");

static void* g_items[] = {
    &g_backend,
    &g_enable,
    &g_model_file,
    nullptr
};

static std::wstring GetPluginLogPath() {
    HMODULE hm = nullptr;
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(&GetPluginLogPath), &hm)) {
        wchar_t module_path[MAX_PATH] = {};
        if (GetModuleFileNameW(hm, module_path, MAX_PATH)) {
            std::filesystem::path p(module_path);
            return (p.parent_path() / L"AI_S-R_ONNX.log").wstring();
        }
    }
    return L"AI_S-R_ONNX.log";
}

static void AppendLogLine(const std::wstring& text) {
    (void)text;
}

static void DebugOut(const wchar_t* text) {
#ifdef _DEBUG
    OutputDebugStringW(text ? text : L"");
    OutputDebugStringW(L"\n");
#else
    (void)text;
#endif
    if (text) AppendLogLine(text);
}

static void DebugOutHotPath(const wchar_t* text) {
#ifdef _DEBUG
    OutputDebugStringW(text ? text : L"");
    OutputDebugStringW(L"\n");
#else
    (void)text;
#endif
    // hot pathでは通常ログファイルへ追記しない
}


static std::wstring TrimNullTerminated(const wchar_t* s) {
    if (!s) return {};
    return std::wstring(s);
}

static std::wstring GetConfiguredModelPath() {
    return TrimNullTerminated(g_model_file.value);
}

static bool FileExists(const std::wstring& path) {
    DWORD attr = GetFileAttributesW(path.c_str());
    return attr != INVALID_FILE_ATTRIBUTES && !(attr & FILE_ATTRIBUTE_DIRECTORY);
}

static std::wstring HResultErrorToString(const winrt::hresult_error& e) {
    std::wstring msg = L"HRESULT 0x";
    wchar_t buf[32] = {};
    swprintf_s(buf, L"%08X", static_cast<uint32_t>(e.code().value));
    msg += buf;
    if (!e.message().empty()) {
        msg += L": ";
        msg += e.message().c_str();
    }
    return msg;
}


// Lookup tables for lossless 8-bit input normalization and FP16 output restore.
struct ScalarLUTs {
    float f32[256];
    uint16_t f16[256];
    uint8_t half_to_u8[65536];
    ScalarLUTs() {
        constexpr float kInv255 = 1.0f / 255.0f;
        for (int i = 0; i < 256; ++i) {
            const float v = static_cast<float>(i) * kInv255;
            f32[i] = v;
            f16[i] = FloatToHalfBits(v);
        }
        for (int i = 0; i < 65536; ++i) {
            float v = HalfBitsToFloat(static_cast<uint16_t>(i));
            v = std::clamp(v, 0.0f, 1.0f) * 255.0f;
            half_to_u8[i] = static_cast<uint8_t>(v + 0.5f);
        }
    }
};

static const ScalarLUTs& GetScalarLUTs() {
    static const ScalarLUTs luts;
    return luts;
}

static inline uint8_t FastClampToU8(float v) {
    if (v <= 0.0f) return 0;
    if (v >= 1.0f) return 255;
    return static_cast<uint8_t>(v * 255.0f + 0.5f);
}

static inline float Clamp01(float v) {
    if (v <= 0.0f) return 0.0f;
    if (v >= 1.0f) return 1.0f;
    return v;
}

static inline float PixelLuma01(const PIXEL_RGBA& p) {
    return (0.299f * static_cast<float>(p.r) + 0.587f * static_cast<float>(p.g) + 0.114f * static_cast<float>(p.b)) * (1.0f / 255.0f);
}

static inline float PixelCb01(const PIXEL_RGBA& p) {
    const float y = PixelLuma01(p);
    const float b = static_cast<float>(p.b) * (1.0f / 255.0f);
    return Clamp01((b - y) / 1.772f + 0.5f);
}

static inline float PixelCr01(const PIXEL_RGBA& p) {
    const float y = PixelLuma01(p);
    const float r = static_cast<float>(p.r) * (1.0f / 255.0f);
    return Clamp01((r - y) / 1.402f + 0.5f);
}

static int ChooseMinRowsPerTask(int w, int h) {
    const int pixels = w * h;
    if (pixels <= 640 * 480) return 24;
    if (pixels <= 960 * 540) return 32;
    if (pixels <= 1280 * 720) return 40;
    if (pixels <= 1920 * 1080) return 48;
    return 64;
}

// Small persistent row-based thread pool to avoid per-frame thread creation overhead.
class RowThreadPool {
public:
    static RowThreadPool& Instance() {
        static RowThreadPool pool;
        return pool;
    }

    template<class Fn>
    void ParallelForRows(int rows, int min_rows_per_task, Fn&& fn) {
        if (rows <= 0) return;
        if (worker_count_ <= 1 || rows < min_rows_per_task * 2) {
            fn(0, rows);
            return;
        }

        std::function<void(int, int)> task = std::forward<Fn>(fn);
        {
            std::unique_lock<std::mutex> lock(mutex_);
            current_task_ = std::move(task);
            total_rows_ = rows;
            next_row_ = 0;
            chunk_rows_ = std::max(min_rows_per_task, rows / static_cast<int>(worker_count_ * 2));
            active_workers_ = worker_count_ - 1;
            generation_++;
        }
        cv_start_.notify_all();

        RunChunks(current_task_, rows, chunk_rows_);

        std::unique_lock<std::mutex> lock(mutex_);
        cv_done_.wait(lock, [&]() { return active_workers_ == 0; });
        current_task_ = nullptr;
    }

private:
    RowThreadPool() {
        unsigned hc = std::thread::hardware_concurrency();
        worker_count_ = hc > 1 ? hc : 1;
        if (worker_count_ > 8) worker_count_ = 8;
        if (worker_count_ <= 1) return;
        threads_.reserve(worker_count_ - 1);
        for (unsigned i = 0; i + 1 < worker_count_; ++i) {
            threads_.emplace_back([this]() { ThreadMain(); });
        }
    }

    ~RowThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            generation_++;
        }
        cv_start_.notify_all();
        for (auto& t : threads_) {
            if (t.joinable()) t.join();
        }
    }

    void ThreadMain() {
        uint64_t seen_generation = 0;
        for (;;) {
            std::function<void(int, int)> task;
            int rows = 0;
            int chunk = 0;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_start_.wait(lock, [&]() { return stopping_ || generation_ != seen_generation; });
                if (stopping_) return;
                seen_generation = generation_;
                task = current_task_;
                rows = total_rows_;
                chunk = chunk_rows_;
            }
            RunChunks(task, rows, chunk);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (--active_workers_ == 0) cv_done_.notify_one();
            }
        }
    }

    void RunChunks(const std::function<void(int, int)>& task, int rows, int chunk) {
        for (;;) {
            int begin = 0;
            int end = 0;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                begin = next_row_;
                if (begin >= rows) break;
                end = std::min(rows, begin + chunk);
                next_row_ = end;
            }
            task(begin, end);
        }
    }

private:
    unsigned worker_count_ = 1;
    std::vector<std::thread> threads_;
    std::mutex mutex_;
    std::condition_variable cv_start_;
    std::condition_variable cv_done_;
    std::function<void(int, int)> current_task_;
    int total_rows_ = 0;
    int next_row_ = 0;
    int chunk_rows_ = 0;
    unsigned active_workers_ = 0;
    uint64_t generation_ = 0;
    bool stopping_ = false;
};


static uint16_t FloatToHalfBits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x007fffffu;

    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant |= 0x00800000u;
        const uint32_t shifted = mant >> static_cast<uint32_t>(1 - exp);
        return static_cast<uint16_t>(sign | ((shifted + 0x00001000u) >> 13));
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u | (mant ? 0x0200u : 0u));
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | ((mant + 0x00001000u) >> 13));
}

static float HalfBitsToFloat(uint16_t value) {
    const uint32_t sign = (static_cast<uint32_t>(value & 0x8000u)) << 16;
    uint32_t exp = (value >> 10) & 0x1fu;
    uint32_t mant = value & 0x03ffu;
    uint32_t bits = 0;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03ffu;
            bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }

    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}
static std::wstring GetPluginDirectoryPath() {
    HMODULE hm = nullptr;
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(&GetPluginDirectoryPath), &hm)) {
        wchar_t module_path[MAX_PATH] = {};
        if (GetModuleFileNameW(hm, module_path, MAX_PATH)) {
            return std::filesystem::path(module_path).parent_path().wstring();
        }
    }
    return std::filesystem::current_path().wstring();
}

static uint64_t HashPathAndTime(const std::wstring& path) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](uint64_t v) {
        for (int i = 0; i < 8; ++i) {
            h ^= static_cast<uint8_t>((v >> (i * 8)) & 0xffu);
            h *= 1099511628211ull;
        }
    };
    for (wchar_t c : path) {
        h ^= static_cast<uint16_t>(c);
        h *= 1099511628211ull;
    }
    WIN32_FILE_ATTRIBUTE_DATA fad{};
    if (GetFileAttributesExW(path.c_str(), GetFileExInfoStandard, &fad)) {
        mix((static_cast<uint64_t>(fad.nFileSizeHigh) << 32) | fad.nFileSizeLow);
        mix((static_cast<uint64_t>(fad.ftLastWriteTime.dwHighDateTime) << 32) | fad.ftLastWriteTime.dwLowDateTime);
    }
    return h;
}

static bool ReadFileBytes(const std::wstring& path, std::vector<uint8_t>& out, std::wstring& error) {
    try {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs) {
            error = L"モデルファイルを開けません";
            return false;
        }
        ifs.seekg(0, std::ios::end);
        std::streamoff size = ifs.tellg();
        if (size < 0) {
            error = L"モデルサイズ取得に失敗しました";
            return false;
        }
        ifs.seekg(0, std::ios::beg);
        out.resize(static_cast<size_t>(size));
        if (!out.empty()) {
            ifs.read(reinterpret_cast<char*>(out.data()), size);
            if (!ifs) {
                error = L"モデル読み込みに失敗しました";
                out.clear();
                return false;
            }
        }
        return true;
    } catch (...) {
        error = L"モデル読み込み中に例外が発生しました";
        out.clear();
        return false;
    }
}

static bool WriteFileBytes(const std::wstring& path, const std::vector<uint8_t>& data, std::wstring& error) {
    try {
        std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
        if (!ofs) {
            error = L"再焼成モデルを書き込めません";
            return false;
        }
        if (!data.empty()) {
            ofs.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
            if (!ofs) {
                error = L"再焼成モデルの書き込みに失敗しました";
                return false;
            }
        }
        return true;
    } catch (...) {
        error = L"再焼成モデル書き込み中に例外が発生しました";
        return false;
    }
}

static bool ReadProtoVarint(const uint8_t* data, size_t size, size_t& pos, uint64_t& value) {
    value = 0;
    int shift = 0;
    while (pos < size && shift <= 63) {
        const uint8_t b = data[pos++];
        value |= static_cast<uint64_t>(b & 0x7fu) << shift;
        if ((b & 0x80u) == 0) return true;
        shift += 7;
    }
    return false;
}

static void AppendProtoVarint(std::vector<uint8_t>& out, uint64_t value) {
    do {
        uint8_t b = static_cast<uint8_t>(value & 0x7fu);
        value >>= 7;
        if (value) b |= 0x80u;
        out.push_back(b);
    } while (value);
}

static bool ExtractOpsetDomain(const uint8_t* data, size_t size, std::string& domain, uint64_t& version) {
    domain.clear();
    version = 0;
    size_t pos = 0;
    while (pos < size) {
        uint64_t key = 0;
        if (!ReadProtoVarint(data, size, pos, key)) return false;
        const uint32_t field = static_cast<uint32_t>(key >> 3);
        const uint32_t wire = static_cast<uint32_t>(key & 7u);
        if (wire == 0) {
            uint64_t v = 0;
            if (!ReadProtoVarint(data, size, pos, v)) return false;
            if (field == 2) version = v;
        } else if (wire == 2) {
            uint64_t len = 0;
            if (!ReadProtoVarint(data, size, pos, len)) return false;
            if (len > size - pos) return false;
            if (field == 1) domain.assign(reinterpret_cast<const char*>(data + pos), static_cast<size_t>(len));
            pos += static_cast<size_t>(len);
        } else if (wire == 1) {
            if (size - pos < 8) return false;
            pos += 8;
        } else if (wire == 5) {
            if (size - pos < 4) return false;
            pos += 4;
        } else {
            return false;
        }
    }
    return true;
}

static bool BuildWinMLCompatibleModelBytes(const std::wstring& model_path, std::vector<uint8_t>& out, bool& changed, std::wstring& error) {
    std::vector<uint8_t> src;
    if (!ReadFileBytes(model_path, src, error)) return false;
    if (src.empty()) {
        error = L"ONNXが空です";
        return false;
    }

    out.clear();
    out.reserve(src.size());
    size_t pos = 0;
    changed = false;
    bool kept_default = false;

    while (pos < src.size()) {
        const size_t field_start = pos;
        uint64_t key = 0;
        if (!ReadProtoVarint(src.data(), src.size(), pos, key)) {
            error = L"ONNX解析失敗: key";
            return false;
        }
        const uint32_t field = static_cast<uint32_t>(key >> 3);
        const uint32_t wire = static_cast<uint32_t>(key & 7u);

        if (wire == 0) {
            uint64_t v = 0;
            if (!ReadProtoVarint(src.data(), src.size(), pos, v)) {
                error = L"ONNX解析失敗: varint";
                return false;
            }
            out.insert(out.end(), src.begin() + field_start, src.begin() + pos);
            continue;
        }
        if (wire == 1) {
            if (src.size() - pos < 8) {
                error = L"ONNX解析失敗: fixed64";
                return false;
            }
            pos += 8;
            out.insert(out.end(), src.begin() + field_start, src.begin() + pos);
            continue;
        }
        if (wire == 5) {
            if (src.size() - pos < 4) {
                error = L"ONNX解析失敗: fixed32";
                return false;
            }
            pos += 4;
            out.insert(out.end(), src.begin() + field_start, src.begin() + pos);
            continue;
        }
        if (wire != 2) {
            error = L"ONNX解析失敗: unsupported wire";
            return false;
        }

        uint64_t len64 = 0;
        if (!ReadProtoVarint(src.data(), src.size(), pos, len64)) {
            error = L"ONNX解析失敗: length";
            return false;
        }
        if (len64 > src.size() - pos) {
            error = L"ONNX解析失敗: truncated field";
            return false;
        }
        const size_t payload_pos = pos;
        const size_t field_end = payload_pos + static_cast<size_t>(len64);
        pos = field_end;

        if (field == 8) {
            std::string domain;
            uint64_t version = 0;
            if (!ExtractOpsetDomain(src.data() + payload_pos, static_cast<size_t>(len64), domain, version)) {
                error = L"ONNX解析失敗: opset_import";
                return false;
            }
            if (domain.empty()) {
                if (!kept_default) {
                    out.insert(out.end(), src.begin() + field_start, src.begin() + field_end);
                    kept_default = true;
                } else {
                    changed = true;
                }
            } else {
                changed = true;
            }
        } else {
            out.insert(out.end(), src.begin() + field_start, src.begin() + field_end);
        }
    }

    if (!changed) {
        out = std::move(src);
    }
    return true;
}

static bool LoadLearningModelPossiblyAbsorbed(const std::wstring& model_path, ml::LearningModel& model, std::vector<uint8_t>& absorbed_bytes, std::wstring& error) {
    absorbed_bytes.clear();
    bool changed = false;
    if (!BuildWinMLCompatibleModelBytes(model_path, absorbed_bytes, changed, error)) {
        return false;
    }

    try {
        if (!changed) {
            model = ml::LearningModel::LoadFromFilePath(model_path);
            return true;
        }

        DebugOut(L"WinML absorb model: in-memory patch");
        wss::InMemoryRandomAccessStream stream;
        wss::DataWriter writer(stream.GetOutputStreamAt(0));
        writer.WriteBytes(winrt::array_view<const uint8_t>(absorbed_bytes));
        writer.StoreAsync().get();
        writer.FlushAsync().get();
        stream.Seek(0);
        auto stream_ref = wss::RandomAccessStreamReference::CreateFromStream(stream);
        model = ml::LearningModel::LoadFromStream(stream_ref);
        return true;
    } catch (const winrt::hresult_error& e) {
        error = HResultErrorToString(e);
        return false;
    }
}



enum class ModelMode {
    Unknown = 0,
    SISR,
    VSR,
};

static const wchar_t* ModelModeName(ModelMode mode) {
    switch (mode) {
    case ModelMode::SISR: return L"SISR";
    case ModelMode::VSR: return L"VSR";
    default: return L"Unknown";
    }
}

struct ModelSpec {
    winrt::hstring input_name;
    winrt::hstring output_name;
    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;
    int64_t in_c = 0;
    int64_t out_c = 0;
    int64_t channels_per_frame = 0;
    int64_t frame_count = 1;
    ml::TensorKind input_kind = ml::TensorKind::Undefined;
    ml::TensorKind output_kind = ml::TensorKind::Undefined;
    ModelMode mode = ModelMode::Unknown;
    bool valid = false;
};

struct PackedFrameCache {
    std::vector<float> f32;
    std::vector<uint16_t> f16;
};

struct AdapterInfo {
    ComPtr<IDXGIAdapter1> adapter;
    std::wstring name;
    LUID luid{};
};

static std::wstring TrimAdapterName(const wchar_t* s) {
    if (!s) return {};
    std::wstring name(s);
    while (!name.empty() && (name.back() == L' ' || name.back() == L'\0')) {
        name.pop_back();
    }
    return name;
}

static std::vector<AdapterInfo> EnumerateHardwareAdapters() {
    std::vector<AdapterInfo> adapters;
    ComPtr<IDXGIFactory6> factory6;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory6)))) {
        ComPtr<IDXGIFactory1> factory1;
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory1)))) {
            return adapters;
        }
        for (UINT i = 0;; ++i) {
            ComPtr<IDXGIAdapter1> adapter;
            if (factory1->EnumAdapters1(i, &adapter) == DXGI_ERROR_NOT_FOUND) break;
            DXGI_ADAPTER_DESC1 desc{};
            if (FAILED(adapter->GetDesc1(&desc))) continue;
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
            adapters.push_back(AdapterInfo{ adapter, TrimAdapterName(desc.Description), desc.AdapterLuid });
        }
        return adapters;
    }

    for (UINT i = 0;; ++i) {
        ComPtr<IDXGIAdapter1> adapter;
        if (factory6->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter)) == DXGI_ERROR_NOT_FOUND) break;
        DXGI_ADAPTER_DESC1 desc{};
        if (FAILED(adapter->GetDesc1(&desc))) continue;
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;

        bool duplicate = false;
        for (const auto& existing : adapters) {
            if (existing.luid.HighPart == desc.AdapterLuid.HighPart &&
                existing.luid.LowPart == desc.AdapterLuid.LowPart) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) {
            adapters.push_back(AdapterInfo{ adapter, TrimAdapterName(desc.Description), desc.AdapterLuid });
        }
    }
    return adapters;
}

static std::wstring FindAdapterNameByLuid(const LUID& luid) {
    const auto adapters = EnumerateHardwareAdapters();
    for (const auto& info : adapters) {
        if (info.luid.HighPart == luid.HighPart && info.luid.LowPart == luid.LowPart) {
            return info.name;
        }
    }
    return {};
}

static void SetWideLabel(wchar_t* dst, size_t dst_count, const std::wstring& text) {
    if (!dst || dst_count == 0) return;
    wcsncpy_s(dst, dst_count, text.c_str(), _TRUNCATE);
}

static void UpdateBackendLabels() {
    SetWideLabel(g_backend_label_auto, std::size(g_backend_label_auto), L"GPU優先(自動)");
    SetWideLabel(g_backend_label_gpu0, std::size(g_backend_label_gpu0), L"GPU 0");
    SetWideLabel(g_backend_label_gpu1, std::size(g_backend_label_gpu1), L"GPU 1");
    SetWideLabel(g_backend_label_cpu, std::size(g_backend_label_cpu), L"CPU固定");

    const auto adapters = EnumerateHardwareAdapters();
    if (!adapters.empty()) {
        SetWideLabel(g_backend_label_auto, std::size(g_backend_label_auto),
                     L"GPU優先(自動): " + adapters[0].name);
        SetWideLabel(g_backend_label_gpu0, std::size(g_backend_label_gpu0),
                     L"GPU 0: " + adapters[0].name);
    }
    if (adapters.size() >= 2) {
        SetWideLabel(g_backend_label_gpu1, std::size(g_backend_label_gpu1),
                     L"GPU 1: " + adapters[1].name);
    }
}

static wgdx11::IDirect3DDevice CreateInteropDeviceFromAdapter(const AdapterInfo& info, std::wstring& error) {
    UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
    D3D_FEATURE_LEVEL levels[] = {
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0
    };
    D3D_FEATURE_LEVEL actual_level = D3D_FEATURE_LEVEL_10_0;

    ComPtr<ID3D11Device> d3d_device;
    ComPtr<ID3D11DeviceContext> d3d_context;
    HRESULT hr = D3D11CreateDevice(
        info.adapter.Get(),
        D3D_DRIVER_TYPE_UNKNOWN,
        nullptr,
        flags,
        levels,
        static_cast<UINT>(std::size(levels)),
        D3D11_SDK_VERSION,
        &d3d_device,
        &actual_level,
        &d3d_context);

    if (FAILED(hr) || !d3d_device) {
        error = L"D3D11デバイス作成失敗: " + info.name;
        return nullptr;
    }

    ComPtr<IDXGIDevice> dxgi_device;
    hr = d3d_device.As(&dxgi_device);
    if (FAILED(hr) || !dxgi_device) {
        error = L"IDXGIDevice取得失敗: " + info.name;
        return nullptr;
    }

    winrt::com_ptr<IInspectable> inspectable;
    hr = CreateDirect3D11DeviceFromDXGIDevice(dxgi_device.Get(), inspectable.put());
    if (FAILED(hr) || !inspectable) {
        error = L"WinRT D3D11デバイス作成失敗: " + info.name;
        return nullptr;
    }

    try {
        return inspectable.as<wgdx11::IDirect3DDevice>();
    } catch (const winrt::hresult_error&) {
        error = L"IDirect3DDevice変換失敗: " + info.name;
        return nullptr;
    }
}

// WinML execution engine. Tensor I/O path is kept unchanged for stability.
class WinMLEngine {
public:
    bool Load(const std::wstring& model_path, int backend_mode, std::wstring& error) {
        Reset();
        try {
            static std::once_flag init_flag;
            std::call_once(init_flag, []() {
                winrt::init_apartment(winrt::apartment_type::multi_threaded);
            });

            DebugOut((L"Load model: " + model_path).c_str());
            if (!LoadLearningModelPossiblyAbsorbed(model_path, model_, absorbed_model_bytes_, error)) {
                DebugOut((L"LoadLearningModel failed: " + error).c_str());
                Reset();
                return false;
            }
            model_path_ = model_path;
            model_spec_ = InspectModel(model_);
            {
                std::wstringstream ss;
                ss << L"InspectModel path=" << model_path_
                   << L" in_shape=";
                for (size_t i = 0; i < model_spec_.input_shape.size(); ++i) {
                    if (i) ss << L"x";
                    ss << model_spec_.input_shape[i];
                }
                ss << L" out_shape=";
                for (size_t i = 0; i < model_spec_.output_shape.size(); ++i) {
                    if (i) ss << L"x";
                    ss << model_spec_.output_shape[i];
                }
                ss << L" in_c=" << model_spec_.in_c
                   << L" out_c=" << model_spec_.out_c
                   << L" valid=" << (model_spec_.valid ? 1 : 0);
                DebugOut(ss.str().c_str());
            }
            if (!model_spec_.valid) {
                error = L"対応外モデル: float32/float16 / 1入力1出力 / 4次元NCHWのみ対応";
                DebugOut((L"InspectModel rejected: " + error).c_str());
                Reset();
                return false;
            }

            if (backend_mode == 3) {
                device_ = ml::LearningModelDevice(ml::LearningModelDeviceKind::Cpu);
                backend_name_ = L"WinML CPU";
                selected_device_name_ = L"CPU";
                ml::LearningModelSessionOptions options;
                options.CloseModelOnSessionCreation(true);
                session_ = ml::LearningModelSession(model_, device_, options);
            } else if (backend_mode == 1 || backend_mode == 2) {
                const int gpu_index = backend_mode - 1;
                const auto adapters = EnumerateHardwareAdapters();
                if (gpu_index < 0 || gpu_index >= static_cast<int>(adapters.size())) {
                    error = (gpu_index == 0) ? L"GPU 0 が見つかりません" : L"GPU 1 が見つかりません";
                    Reset();
                    return false;
                }

                auto d3d_device = CreateInteropDeviceFromAdapter(adapters[gpu_index], error);
                if (!d3d_device) {
                    Reset();
                    return false;
                }

                device_ = ml::LearningModelDevice::CreateFromDirect3D11Device(d3d_device);
                backend_name_ = L"WinML GPU " + std::to_wstring(gpu_index);
                selected_device_name_ = adapters[gpu_index].name;
                ml::LearningModelSessionOptions options;
                options.CloseModelOnSessionCreation(true);
                session_ = ml::LearningModelSession(model_, device_, options);
            } else {
                try {
                    device_ = ml::LearningModelDevice(ml::LearningModelDeviceKind::DirectXHighPerformance);
                    selected_device_name_.clear();
                    try {
                        const auto adapter_id = device_.AdapterId();
                        LUID luid{};
                        luid.HighPart = adapter_id.HighPart;
                        luid.LowPart = adapter_id.LowPart;
                        selected_device_name_ = FindAdapterNameByLuid(luid);
                    } catch (...) {
                    }

                    backend_name_ = selected_device_name_.empty()
                        ? L"WinML GPU (自動)"
                        : (L"WinML GPU (自動): " + selected_device_name_);
                    ml::LearningModelSessionOptions options;
                    options.CloseModelOnSessionCreation(true);
                    session_ = ml::LearningModelSession(model_, device_, options);
                } catch (const winrt::hresult_error& gpu_e) {
                    DebugOut((L"WinML GPU fallback: " + HResultErrorToString(gpu_e)).c_str());
                    device_ = ml::LearningModelDevice(ml::LearningModelDeviceKind::Cpu);
                    backend_name_ = L"WinML CPU";
                    selected_device_name_ = L"CPU";
                    ml::LearningModelSessionOptions options;
                    options.CloseModelOnSessionCreation(true);
                    session_ = ml::LearningModelSession(model_, device_, options);
                    error = L"GPU初期化失敗";
                }
            }

            if (!selected_device_name_.empty()) {
                DebugOut((L"Selected device: " + selected_device_name_).c_str());
            }

            loaded_ = true;
            return true;
        } catch (const winrt::hresult_error& e) {
            error = HResultErrorToString(e);
            Reset();
            return false;
        } catch (const std::exception& e) {
            error = L"例外: ";
            error += winrt::to_hstring(e.what()).c_str();
            Reset();
            return false;
        }
    }

    void Reset() {
        loaded_ = false;
        model_ = nullptr;
        device_ = nullptr;
        session_ = nullptr;
        model_spec_ = {};
        backend_name_.clear();
        selected_device_name_.clear();
        model_path_.clear();
        absorbed_model_bytes_.clear();
        bound_input_shape_.clear();
        input_tensor_f32_ = nullptr;
        input_tensor_f16_ = nullptr;
        bound_output_shape_.clear();
        output_tensor_f32_ = nullptr;
        output_tensor_f16_ = nullptr;
        output_binding_ready_ = false;
        cached_binding_ = nullptr;
        binding_ready_ = false;
        defer_output_binding_for_eval_ = false;
        packed_input_f32_.clear();
        packed_input_f16_.clear();
        frame_scratch_f32_.clear();
        frame_scratch_f16_.clear();
        last_output_pixels_.clear();
        last_output_width_ = 0;
        last_output_height_ = 0;
        last_window_width_ = 0;
        last_window_height_ = 0;
        last_sisr_input_width_ = 0;
        last_sisr_input_height_ = 0;
        has_last_frame_ = false;
        last_frame_number_ = std::numeric_limits<int>::min();
        last_object_id_ = 0;
        last_effect_id_ = 0;
    }

    bool IsLoaded() const { return loaded_ && session_ != nullptr; }
    const ModelSpec& Spec() const { return model_spec_; }
    const std::wstring& BackendName() const { return backend_name_; }
    const std::wstring& SelectedDeviceName() const { return selected_device_name_; }
    const std::wstring& ModelPath() const { return model_path_; }

    bool RunPixels(const PIXEL_RGBA* src_pixels, int width, int height,
                   int object_frame, int64_t object_id, int64_t effect_id,
                   std::vector<PIXEL_RGBA>& dst_pixels,
                   int& out_width, int& out_height, std::wstring& error) {
        out_width = 0;
        out_height = 0;
        if (!IsLoaded()) {
            error = L"モデル未読込";
            return false;
        }
        if (!src_pixels || width <= 0 || height <= 0) {
            error = L"不正サイズ";
            return false;
        }

        if (model_spec_.mode == ModelMode::SISR) {
            if (last_sisr_input_width_ > 0 && last_sisr_input_height_ > 0 &&
                (last_sisr_input_width_ != width || last_sisr_input_height_ != height)) {
                InvalidateSISRExecutionState();
                std::wstringstream ss;
                ss << L"SISR safe invalidate: input size changed "
                   << last_sisr_input_width_ << L"x" << last_sisr_input_height_
                   << L" -> " << width << L"x" << height;
                DebugOut(ss.str().c_str());
            }
            last_sisr_input_width_ = width;
            last_sisr_input_height_ = height;
        }

        if (model_spec_.mode == ModelMode::VSR && has_last_frame_ &&
            object_id == last_object_id_ && effect_id == last_effect_id_ &&
            object_frame == last_frame_number_ &&
            !last_output_pixels_.empty() && last_output_width_ > 0 && last_output_height_ > 0) {
            dst_pixels = last_output_pixels_;
            out_width = last_output_width_;
            out_height = last_output_height_;
            error.clear();
            DebugOut(L"VSR cache hit: repeated frame reuse");
            return true;
        }

        const auto t0 = std::chrono::steady_clock::now();
        long long ms_shape = 0;
        long long ms_tensor = 0;
        long long ms_window = 0;
        long long ms_pack = 0;
        long long ms_bind = 0;
        long long ms_eval = 0;
        long long ms_lookup = 0;
        long long ms_validate = 0;
        long long ms_unpack = 0;

        const int in_c = static_cast<int>(model_spec_.in_c);
        auto ts = std::chrono::steady_clock::now();
        auto input_shape = ResolveInputShape(width, height);
        if (input_shape.size() != 4) {
            error = L"入力shape解決失敗";
            return false;
        }
        ms_shape = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - ts).count();

        ts = std::chrono::steady_clock::now();
        if (!EnsureInputTensor(input_shape, error)) {
            return false;
        }
        ms_tensor = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - ts).count();

        if (model_spec_.mode == ModelMode::SISR) {
            if (!(in_c == 1 || in_c == 3 || in_c == 4)) {
                error = L"入力チャネル数が1/3/4ではありません";
                return false;
            }
            ts = std::chrono::steady_clock::now();
            if (model_spec_.input_kind == ml::TensorKind::Float16) {
                if (!FillInputTensorFloat16(src_pixels, width, height, in_c, error)) {
                    return false;
                }
            } else {
                if (!FillInputTensorFloat32(src_pixels, width, height, in_c, error)) {
                    return false;
                }
            }
            ms_pack = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - ts).count();
        } else if (model_spec_.mode == ModelMode::VSR) {
            const int frame_count = static_cast<int>(std::max<int64_t>(1, model_spec_.frame_count));
            const int channels_per_frame = static_cast<int>(std::max<int64_t>(1, model_spec_.channels_per_frame));
            if (!((channels_per_frame == 3 || channels_per_frame == 4) && frame_count >= 1)) {
                error = L"VSR入力チャネル構成に未対応です";
                return false;
            }

            ts = std::chrono::steady_clock::now();
            PrepareTemporalWindow(src_pixels, width, height, frame_count, object_frame, object_id, effect_id);
            ms_window = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - ts).count();

            ts = std::chrono::steady_clock::now();
            if (model_spec_.input_kind == ml::TensorKind::Float16) {
                if (!FillPackedInputTensorFloat16(packed_input_f16_, error)) {
                    return false;
                }
            } else {
                if (!FillPackedInputTensorFloat32(packed_input_f32_, error)) {
                    return false;
                }
            }
            ms_pack = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - ts).count();
        } else {
            error = L"対応外モデルです";
            return false;
        }

        ts = std::chrono::steady_clock::now();
        auto guessed_output_shape = GuessOutputShape(input_shape);
        if (!guessed_output_shape.empty()) {
            std::wstring out_err;
            EnsureOutputTensorForShape(guessed_output_shape, out_err);
        }
        if (!EnsureBindings(error)) {
            return false;
        }
        ms_bind = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - ts).count();

        ts = std::chrono::steady_clock::now();
        decltype(session_.Evaluate(cached_binding_, L"AI S-R ONNX")) result{ nullptr };
        try {
            result = session_.Evaluate(cached_binding_, L"AI S-R ONNX");
        } catch (const winrt::hresult_error& e) {
            const std::wstring first_error = HResultErrorToString(e);
            if (!defer_output_binding_for_eval_ && model_spec_.mode == ModelMode::SISR) {
                DebugOut((L"Evaluate failed, retry without prebound output path=" + model_path_ + L" err=" + first_error).c_str());
                defer_output_binding_for_eval_ = true;
                output_binding_ready_ = false;
                output_tensor_f32_ = nullptr;
                output_tensor_f16_ = nullptr;
                cached_binding_ = nullptr;
                binding_ready_ = false;
                std::wstring retry_bind_error;
                if (!EnsureBindings(retry_bind_error)) {
                    error = retry_bind_error;
                    return false;
                }
                try {
                    result = session_.Evaluate(cached_binding_, L"AI S-R ONNX");
                } catch (const winrt::hresult_error& e2) {
                    error = L"Evaluate失敗: " + HResultErrorToString(e2);
                    DebugOut((L"Evaluate failed after retry path=" + model_path_ + L" err=" + error).c_str());
                    return false;
                }
            } else {
                error = L"Evaluate失敗: " + first_error;
                DebugOut((L"Evaluate failed path=" + model_path_ + L" err=" + error).c_str());
                return false;
            }
        }
        ms_eval = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - ts).count();

        if (model_spec_.output_kind == ml::TensorKind::Float16) {
            ml::TensorFloat16Bit out_tensor{ nullptr };
            std::vector<int64_t> out_shape;
            ts = std::chrono::steady_clock::now();
            if (output_binding_ready_ && output_tensor_f16_) {
                out_tensor = output_tensor_f16_;
                out_shape = bound_output_shape_;
            } else {
                auto outputs = result.Outputs();
                winrt::Windows::Foundation::IInspectable boxed{ nullptr };
                try {
                    boxed = outputs.Lookup(model_spec_.output_name);
                } catch (const winrt::hresult_error&) {
                    error = L"出力名が見つかりません";
                    return false;
                }
                out_tensor = boxed.try_as<ml::TensorFloat16Bit>();
                if (!out_tensor) {
                    error = L"出力テンソルがTensorFloat16Bitではありません";
                    return false;
                }
                out_shape = ToStdVector(out_tensor.Shape());
                EnsureOutputTensorForShape(out_shape, error);
            }
            ms_lookup = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - ts).count();
            ts = std::chrono::steady_clock::now();
            if (!ValidateOutputShape(out_shape, out_width, out_height, error)) {
                return false;
            }
            ms_validate = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - ts).count();
            ts = std::chrono::steady_clock::now();
            if (!UnpackOutputTensorFloat16(out_tensor, out_width, out_height, static_cast<int>(out_shape[1]), dst_pixels, error,
                                          src_pixels, width, height)) {
                return false;
            }
            ms_unpack = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - ts).count();
        } else {
            ml::TensorFloat out_tensor{ nullptr };
            std::vector<int64_t> out_shape;
            ts = std::chrono::steady_clock::now();
            if (output_binding_ready_ && output_tensor_f32_) {
                out_tensor = output_tensor_f32_;
                out_shape = bound_output_shape_;
            } else {
                auto outputs = result.Outputs();
                winrt::Windows::Foundation::IInspectable boxed{ nullptr };
                try {
                    boxed = outputs.Lookup(model_spec_.output_name);
                } catch (const winrt::hresult_error&) {
                    error = L"出力名が見つかりません";
                    return false;
                }
                out_tensor = boxed.try_as<ml::TensorFloat>();
                if (!out_tensor) {
                    error = L"出力テンソルがTensorFloatではありません";
                    return false;
                }
                out_shape = ToStdVector(out_tensor.Shape());
                EnsureOutputTensorForShape(out_shape, error);
            }
            ms_lookup = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - ts).count();
            ts = std::chrono::steady_clock::now();
            if (!ValidateOutputShape(out_shape, out_width, out_height, error)) {
                return false;
            }
            ms_validate = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - ts).count();
            ts = std::chrono::steady_clock::now();
            if (!UnpackOutputTensorFloat32(out_tensor, out_width, out_height, static_cast<int>(out_shape[1]), dst_pixels, error,
                                          src_pixels, width, height)) {
                return false;
            }
            ms_unpack = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - ts).count();
        }

        if (model_spec_.mode == ModelMode::VSR) {
            last_output_pixels_ = dst_pixels;
            last_output_width_ = out_width;
            last_output_height_ = out_height;
        }
        const auto t1 = std::chrono::steady_clock::now();
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        {
            std::wstringstream ss;
            ss << L"RunPixels mode=" << ModelModeName(model_spec_.mode)
               << L" total=" << ms
               << L" shape=" << ms_shape
               << L" tensor=" << ms_tensor
               << L" window=" << ms_window
               << L" pack=" << ms_pack
               << L" bind=" << ms_bind
               << L" eval=" << ms_eval
               << L" outbind=" << (output_binding_ready_ ? 1 : 0)
               << L" deferred_out=" << (defer_output_binding_for_eval_ ? 1 : 0)
               << L" lookup=" << ms_lookup
               << L" validate=" << ms_validate
               << L" unpack=" << ms_unpack;
            DebugOutHotPath(ss.str().c_str());
        }
        error.clear();
        return true;
    }

private:
    static ModelSpec InspectModel(const ml::LearningModel& model) {
        ModelSpec spec;
        auto ins = model.InputFeatures();
        auto outs = model.OutputFeatures();
        if (ins.Size() != 1 || outs.Size() != 1) return spec;

        auto in_base = ins.GetAt(0);
        auto out_base = outs.GetAt(0);
        if (in_base.Kind() != ml::LearningModelFeatureKind::Tensor ||
            out_base.Kind() != ml::LearningModelFeatureKind::Tensor) {
            return spec;
        }

        auto in_desc = in_base.as<ml::TensorFeatureDescriptor>();
        auto out_desc = out_base.as<ml::TensorFeatureDescriptor>();
        const auto in_kind = in_desc.TensorKind();
        const auto out_kind = out_desc.TensorKind();
        if (!((in_kind == ml::TensorKind::Float || in_kind == ml::TensorKind::Float16) &&
              (out_kind == ml::TensorKind::Float || out_kind == ml::TensorKind::Float16))) {
            return spec;
        }

        spec.input_name = in_base.Name();
        spec.input_kind = in_kind;
        spec.output_kind = out_kind;
        spec.output_name = out_base.Name();
        spec.input_shape = ToStdVector(in_desc.Shape());
        spec.output_shape = ToStdVector(out_desc.Shape());
        if (spec.input_shape.size() != 4 || spec.output_shape.size() != 4) {
            return spec;
        }

        spec.in_c = spec.input_shape[1];
        spec.out_c = spec.output_shape[1];
        if (spec.in_c <= 0) return spec;

        if (spec.out_c <= 0 && (spec.in_c == 1 || spec.in_c == 3 || spec.in_c == 4)) {
            // Some SISR ONNX export paths leave output channel symbolic in the graph metadata.
            // In practice these models usually preserve channel count, so accept them here.
            spec.out_c = spec.in_c;
        }
        if (spec.out_c <= 0) return spec;

        if (spec.in_c == 1 || spec.in_c == 3 || spec.in_c == 4) {
            spec.mode = ModelMode::SISR;
            spec.frame_count = 1;
            spec.channels_per_frame = spec.in_c;
        } else if ((spec.in_c % 3) == 0) {
            spec.mode = ModelMode::VSR;
            spec.channels_per_frame = 3;
            spec.frame_count = spec.in_c / 3;
        } else if ((spec.in_c % 4) == 0) {
            spec.mode = ModelMode::VSR;
            spec.channels_per_frame = 4;
            spec.frame_count = spec.in_c / 4;
        } else {
            return spec;
        }

        if (!(spec.out_c == 1 || spec.out_c == 3 || spec.out_c == 4)) return spec;
        spec.valid = true;
        return spec;
    }

    static std::vector<int64_t> ToStdVector(const wfc::IVectorView<int64_t>& view) {
        std::vector<int64_t> v;
        v.reserve(view.Size());
        for (uint32_t i = 0; i < view.Size(); ++i) v.push_back(view.GetAt(i));
        return v;
    }

    std::vector<int64_t> ResolveInputShape(int w, int h) const {
        auto s = model_spec_.input_shape;
        if (s.size() != 4) return {};
        if (s[0] <= 0) s[0] = 1;
        if (s[1] <= 0) s[1] = model_spec_.in_c;
        if (s[2] <= 0) s[2] = h;
        if (s[3] <= 0) s[3] = w;
        return s;
    }

    std::vector<int64_t> GuessOutputShape(const std::vector<int64_t>& input_shape) const {
        if (model_spec_.output_shape.size() != 4 || input_shape.size() != 4) return {};
        auto out = model_spec_.output_shape;
        bool ok = true;
        for (int i = 0; i < 4; ++i) {
            if (out[i] > 0) continue;
            if (i == 0) out[i] = 1;
            else if (i == 1) out[i] = model_spec_.out_c;
            else if (model_spec_.input_shape[i] > 0 && input_shape[i] > 0 && out[i] == model_spec_.output_shape[i]) {
                ok = false;
            }
        }
        auto scale_dim = [&](int idx) -> int64_t {
            const auto in_decl = model_spec_.input_shape[idx];
            const auto out_decl = model_spec_.output_shape[idx];
            const auto in_now = input_shape[idx];
            if (in_decl > 0 && out_decl > 0 && in_now > 0) {
                return (out_decl * in_now) / in_decl;
            }
            return -1;
        };
        if (out[2] <= 0) {
            auto v = scale_dim(2);
            if (v > 0) out[2] = v;
            else ok = false;
        }
        if (out[3] <= 0) {
            auto v = scale_dim(3);
            if (v > 0) out[3] = v;
            else ok = false;
        }
        if (out[0] <= 0) out[0] = 1;
        if (out[1] <= 0) out[1] = model_spec_.out_c;
        if (!ok) return {};
        return out;
    }

    bool EnsureInputTensor(const std::vector<int64_t>& input_shape, std::wstring& error) {
        if (input_shape == bound_input_shape_) return true;

        try {
            if (model_spec_.input_kind == ml::TensorKind::Float16) {
                input_tensor_f16_ = ml::TensorFloat16Bit::Create(input_shape);
                input_tensor_f32_ = nullptr;
            } else {
                input_tensor_f32_ = ml::TensorFloat::Create(input_shape);
                input_tensor_f16_ = nullptr;
            }
            bound_input_shape_ = input_shape;
            cached_binding_ = nullptr;
            binding_ready_ = false;
            return true;
        } catch (const winrt::hresult_error& e) {
            error = L"入力テンソル作成失敗: " + HResultErrorToString(e);
            bound_input_shape_.clear();
            input_tensor_f32_ = nullptr;
            input_tensor_f16_ = nullptr;
            return false;
        }
    }

    // Cache input/output binding.
    bool EnsureBindings(std::wstring& error) {
        try {
            if (!cached_binding_) {
                cached_binding_ = ml::LearningModelBinding(session_);
                binding_ready_ = false;
                output_binding_ready_ = false;
            }
            if (!binding_ready_) {
                if (model_spec_.input_kind == ml::TensorKind::Float16) {
                    cached_binding_.Bind(model_spec_.input_name, input_tensor_f16_);
                } else {
                    cached_binding_.Bind(model_spec_.input_name, input_tensor_f32_);
                }
                binding_ready_ = true;
            }
            if (!defer_output_binding_for_eval_ && !bound_output_shape_.empty() && !output_binding_ready_) {
                if (model_spec_.output_kind == ml::TensorKind::Float16) {
                    if (!output_tensor_f16_) output_tensor_f16_ = ml::TensorFloat16Bit::Create(bound_output_shape_);
                    cached_binding_.Bind(model_spec_.output_name, output_tensor_f16_);
                } else {
                    if (!output_tensor_f32_) output_tensor_f32_ = ml::TensorFloat::Create(bound_output_shape_);
                    cached_binding_.Bind(model_spec_.output_name, output_tensor_f32_);
                }
                output_binding_ready_ = true;
            }
            return true;
        } catch (const winrt::hresult_error& e) {
            error = L"入出力バインド作成失敗: " + HResultErrorToString(e);
            cached_binding_ = nullptr;
            binding_ready_ = false;
            output_binding_ready_ = false;
            return false;
        }
    }

    bool EnsureOutputTensorForShape(const std::vector<int64_t>& out_shape, std::wstring& error) {
        try {
            if (out_shape == bound_output_shape_) return true;
            bound_output_shape_ = out_shape;
            output_tensor_f32_ = nullptr;
            output_tensor_f16_ = nullptr;
            output_binding_ready_ = false;
            if (cached_binding_) {
                if (model_spec_.output_kind == ml::TensorKind::Float16) {
                    output_tensor_f16_ = ml::TensorFloat16Bit::Create(bound_output_shape_);
                    cached_binding_.Bind(model_spec_.output_name, output_tensor_f16_);
                } else {
                    output_tensor_f32_ = ml::TensorFloat::Create(bound_output_shape_);
                    cached_binding_.Bind(model_spec_.output_name, output_tensor_f32_);
                }
                output_binding_ready_ = true;
            }
            return true;
        } catch (const winrt::hresult_error& e) {
            error = L"出力テンソル作成失敗: " + HResultErrorToString(e);
            bound_output_shape_.clear();
            output_tensor_f32_ = nullptr;
            output_tensor_f16_ = nullptr;
            output_binding_ready_ = false;
            return false;
        }
    }





    static bool ValidateOutputShape(const std::vector<int64_t>& out_shape, int& out_width, int& out_height, std::wstring& error) {
        if (out_shape.size() != 4) {
            error = L"出力shape想定外";
            return false;
        }

        const int out_n = static_cast<int>(out_shape[0]);
        const int out_c = static_cast<int>(out_shape[1]);
        const int out_h = static_cast<int>(out_shape[2]);
        const int out_w = static_cast<int>(out_shape[3]);
        if (out_n != 1 || !(out_c == 1 || out_c == 3 || out_c == 4) || out_w <= 0 || out_h <= 0) {
            error = L"出力shape想定外";
            return false;
        }

        const uint64_t max_pixels = 7680ull * 4320ull * 4ull;
        const uint64_t out_pixels = static_cast<uint64_t>(out_w) * static_cast<uint64_t>(out_h);
        if (out_pixels == 0 || out_pixels > max_pixels) {
            error = L"出力サイズが大きすぎます";
            return false;
        }

        out_width = out_w;
        out_height = out_h;
        return true;
    }

    static bool FillTensorFloatBuffer(const ml::TensorFloat& tensor, const PIXEL_RGBA* src_pixels, int w, int h, int channels, std::wstring& error) {
        try {
            auto ref = tensor.CreateReference();
            auto access = ref.as<IMemoryBufferByteAccess>();
            BYTE* bytes = nullptr;
            UINT32 capacity = 0;
            if (FAILED(access->GetBuffer(&bytes, &capacity)) || !bytes) {
                error = L"TensorFloat バッファ取得失敗";
                return false;
            }
            float* out = reinterpret_cast<float*>(bytes);
            const size_t plane = static_cast<size_t>(w) * static_cast<size_t>(h);
            const size_t plane2 = plane * 2;
            const size_t plane3 = plane * 3;
            const size_t needed = plane * static_cast<size_t>(channels) * sizeof(float);
            if (capacity < needed) {
                error = L"TensorFloat バッファサイズ不足";
                return false;
            }
            const auto& lut = GetScalarLUTs();
            if (channels == 4) {
                auto work = [&](int y0, int y1) {
                    for (int y = y0; y < y1; ++y) {
                        const size_t row_base = static_cast<size_t>(y) * static_cast<size_t>(w);
                        const PIXEL_RGBA* __restrict srow = src_pixels + row_base;
                        float* __restrict rrow = out + row_base;
                        float* __restrict grow = out + plane + row_base;
                        float* __restrict brow = out + plane2 + row_base;
                        float* __restrict arow = out + plane3 + row_base;
                        for (int x = 0; x < w; ++x) {
                            const PIXEL_RGBA& s = srow[x];
                            rrow[x] = lut.f32[s.r];
                            grow[x] = lut.f32[s.g];
                            brow[x] = lut.f32[s.b];
                            arow[x] = lut.f32[s.a];
                        }
                    }
                };
                RowThreadPool::Instance().ParallelForRows(h, ChooseMinRowsPerTask(w, h), work);
            } else if (channels == 3) {
                auto work = [&](int y0, int y1) {
                    for (int y = y0; y < y1; ++y) {
                        const size_t row_base = static_cast<size_t>(y) * static_cast<size_t>(w);
                        const PIXEL_RGBA* __restrict srow = src_pixels + row_base;
                        float* __restrict rrow = out + row_base;
                        float* __restrict grow = out + plane + row_base;
                        float* __restrict brow = out + plane2 + row_base;
                        for (int x = 0; x < w; ++x) {
                            const PIXEL_RGBA& s = srow[x];
                            rrow[x] = lut.f32[s.r];
                            grow[x] = lut.f32[s.g];
                            brow[x] = lut.f32[s.b];
                        }
                    }
                };
                RowThreadPool::Instance().ParallelForRows(h, ChooseMinRowsPerTask(w, h), work);
            } else if (channels == 1) {
                auto work = [&](int y0, int y1) {
                    for (int y = y0; y < y1; ++y) {
                        const size_t row_base = static_cast<size_t>(y) * static_cast<size_t>(w);
                        const PIXEL_RGBA* __restrict srow = src_pixels + row_base;
                        float* __restrict yrow = out + row_base;
                        for (int x = 0; x < w; ++x) {
                            const PIXEL_RGBA& s = srow[x];
                            const int gray = (77 * int(s.r) + 150 * int(s.g) + 29 * int(s.b) + 128) >> 8;
                            yrow[x] = lut.f32[gray];
                        }
                    }
                };
                RowThreadPool::Instance().ParallelForRows(h, ChooseMinRowsPerTask(w, h), work);
            } else {
                error = L"TensorFloat 未対応チャネル数";
                return false;
            }
            return true;
        } catch (const winrt::hresult_error& e) {
            error = L"TensorFloat 書き込み失敗: " + HResultErrorToString(e);
            return false;
        }
    }

    static bool FillTensorFloat16Buffer(const ml::TensorFloat16Bit& tensor, const PIXEL_RGBA* src_pixels, int w, int h, int channels, std::wstring& error) {
        try {
            auto ref = tensor.CreateReference();
            auto access = ref.as<IMemoryBufferByteAccess>();
            BYTE* bytes = nullptr;
            UINT32 capacity = 0;
            if (FAILED(access->GetBuffer(&bytes, &capacity)) || !bytes) {
                error = L"TensorFloat16Bit バッファ取得失敗";
                return false;
            }
            uint16_t* out = reinterpret_cast<uint16_t*>(bytes);
            const size_t plane = static_cast<size_t>(w) * static_cast<size_t>(h);
            const size_t plane2 = plane * 2;
            const size_t plane3 = plane * 3;
            const size_t needed = plane * static_cast<size_t>(channels) * sizeof(uint16_t);
            if (capacity < needed) {
                error = L"TensorFloat16Bit バッファサイズ不足";
                return false;
            }
            const auto& lut = GetScalarLUTs();
            if (channels == 4) {
                auto work = [&](int y0, int y1) {
                    for (int y = y0; y < y1; ++y) {
                        const size_t row_base = static_cast<size_t>(y) * static_cast<size_t>(w);
                        const PIXEL_RGBA* __restrict srow = src_pixels + row_base;
                        uint16_t* __restrict rrow = out + row_base;
                        uint16_t* __restrict grow = out + plane + row_base;
                        uint16_t* __restrict brow = out + plane2 + row_base;
                        uint16_t* __restrict arow = out + plane3 + row_base;
                        for (int x = 0; x < w; ++x) {
                            const PIXEL_RGBA& s = srow[x];
                            rrow[x] = lut.f16[s.r];
                            grow[x] = lut.f16[s.g];
                            brow[x] = lut.f16[s.b];
                            arow[x] = lut.f16[s.a];
                        }
                    }
                };
                RowThreadPool::Instance().ParallelForRows(h, ChooseMinRowsPerTask(w, h), work);
            } else if (channels == 3) {
                auto work = [&](int y0, int y1) {
                    for (int y = y0; y < y1; ++y) {
                        const size_t row_base = static_cast<size_t>(y) * static_cast<size_t>(w);
                        const PIXEL_RGBA* __restrict srow = src_pixels + row_base;
                        uint16_t* __restrict rrow = out + row_base;
                        uint16_t* __restrict grow = out + plane + row_base;
                        uint16_t* __restrict brow = out + plane2 + row_base;
                        for (int x = 0; x < w; ++x) {
                            const PIXEL_RGBA& s = srow[x];
                            rrow[x] = lut.f16[s.r];
                            grow[x] = lut.f16[s.g];
                            brow[x] = lut.f16[s.b];
                        }
                    }
                };
                RowThreadPool::Instance().ParallelForRows(h, ChooseMinRowsPerTask(w, h), work);
            } else if (channels == 1) {
                auto work = [&](int y0, int y1) {
                    for (int y = y0; y < y1; ++y) {
                        const size_t row_base = static_cast<size_t>(y) * static_cast<size_t>(w);
                        const PIXEL_RGBA* __restrict srow = src_pixels + row_base;
                        uint16_t* __restrict yrow = out + row_base;
                        for (int x = 0; x < w; ++x) {
                            const PIXEL_RGBA& s = srow[x];
                            const int gray = (77 * int(s.r) + 150 * int(s.g) + 29 * int(s.b) + 128) >> 8;
                            yrow[x] = lut.f16[gray];
                        }
                    }
                };
                RowThreadPool::Instance().ParallelForRows(h, ChooseMinRowsPerTask(w, h), work);
            } else {
                error = L"TensorFloat16Bit 未対応チャネル数";
                return false;
            }
            return true;
        } catch (const winrt::hresult_error& e) {
            error = L"TensorFloat16Bit 書き込み失敗: " + HResultErrorToString(e);
            return false;
        }
    }

    bool FillInputTensorFloat32(const PIXEL_RGBA* src_pixels, int w, int h, int channels, std::wstring& error) {
        return FillTensorFloatBuffer(input_tensor_f32_, src_pixels, w, h, channels, error);
    }

    bool FillInputTensorFloat16(const PIXEL_RGBA* src_pixels, int w, int h, int channels, std::wstring& error) {
        return FillTensorFloat16Buffer(input_tensor_f16_, src_pixels, w, h, channels, error);
    }

    static void UnpackFromFloatBuffer(const float* src, int w, int h, int channels, std::vector<PIXEL_RGBA>& out_pixels,
                                     const PIXEL_RGBA* src_pixels = nullptr, int src_w = 0, int src_h = 0) {
        const size_t plane = static_cast<size_t>(w) * static_cast<size_t>(h);
        const size_t plane2 = plane * 2;
        const size_t plane3 = plane * 3;
        if (out_pixels.size() != plane) out_pixels.resize(plane);
        if (channels == 4) {
            auto work = [&](int y0, int y1) {
                for (int y = y0; y < y1; ++y) {
                    const size_t row_base = static_cast<size_t>(y) * static_cast<size_t>(w);
                    PIXEL_RGBA* __restrict drow = out_pixels.data() + row_base;
                    const float* __restrict rrow = src + row_base;
                    const float* __restrict grow = src + plane + row_base;
                    const float* __restrict brow = src + plane2 + row_base;
                    const float* __restrict arow = src + plane3 + row_base;
                    for (int x = 0; x < w; ++x) {
                        PIXEL_RGBA& d = drow[x];
                        d.r = FastClampToU8(rrow[x]);
                        d.g = FastClampToU8(grow[x]);
                        d.b = FastClampToU8(brow[x]);
                        d.a = FastClampToU8(arow[x]);
                    }
                }
            };
            RowThreadPool::Instance().ParallelForRows(h, ChooseMinRowsPerTask(w, h), work);
        } else if (channels == 3) {
            auto work = [&](int y0, int y1) {
                for (int y = y0; y < y1; ++y) {
                    const size_t row_base = static_cast<size_t>(y) * static_cast<size_t>(w);
                    PIXEL_RGBA* __restrict drow = out_pixels.data() + row_base;
                    const float* __restrict rrow = src + row_base;
                    const float* __restrict grow = src + plane + row_base;
                    const float* __restrict brow = src + plane2 + row_base;
                    for (int x = 0; x < w; ++x) {
                        PIXEL_RGBA& d = drow[x];
                        d.r = FastClampToU8(rrow[x]);
                        d.g = FastClampToU8(grow[x]);
                        d.b = FastClampToU8(brow[x]);
                        d.a = 255;
                    }
                }
            };
            RowThreadPool::Instance().ParallelForRows(h, ChooseMinRowsPerTask(w, h), work);
        } else if (channels == 1) {
            if (src_pixels && src_w > 0 && src_h > 0) {
                auto sample_chroma = [&](float sx, float sy, bool cb) -> float {
                    sx = std::clamp(sx, 0.0f, static_cast<float>(src_w - 1));
                    sy = std::clamp(sy, 0.0f, static_cast<float>(src_h - 1));
                    const int x0 = static_cast<int>(sx);
                    const int y0 = static_cast<int>(sy);
                    const int x1 = std::min(x0 + 1, src_w - 1);
                    const int y1 = std::min(y0 + 1, src_h - 1);
                    const float fx = sx - static_cast<float>(x0);
                    const float fy = sy - static_cast<float>(y0);
                    const auto& p00 = src_pixels[static_cast<size_t>(y0) * static_cast<size_t>(src_w) + static_cast<size_t>(x0)];
                    const auto& p10 = src_pixels[static_cast<size_t>(y0) * static_cast<size_t>(src_w) + static_cast<size_t>(x1)];
                    const auto& p01 = src_pixels[static_cast<size_t>(y1) * static_cast<size_t>(src_w) + static_cast<size_t>(x0)];
                    const auto& p11 = src_pixels[static_cast<size_t>(y1) * static_cast<size_t>(src_w) + static_cast<size_t>(x1)];
                    const float c00 = cb ? PixelCb01(p00) : PixelCr01(p00);
                    const float c10 = cb ? PixelCb01(p10) : PixelCr01(p10);
                    const float c01 = cb ? PixelCb01(p01) : PixelCr01(p01);
                    const float c11 = cb ? PixelCb01(p11) : PixelCr01(p11);
                    const float c0 = c00 + (c10 - c00) * fx;
                    const float c1 = c01 + (c11 - c01) * fx;
                    return c0 + (c1 - c0) * fy;
                };
                auto work = [&](int y0, int y1) {
                    for (int y = y0; y < y1; ++y) {
                        const size_t row_base = static_cast<size_t>(y) * static_cast<size_t>(w);
                        PIXEL_RGBA* __restrict drow = out_pixels.data() + row_base;
                        const float* __restrict yrow = src + row_base;
                        const float sy = ((static_cast<float>(y) + 0.5f) * static_cast<float>(src_h) / static_cast<float>(h)) - 0.5f;
                        for (int x = 0; x < w; ++x) {
                            const float yv = Clamp01(yrow[x]);
                            const float sx = ((static_cast<float>(x) + 0.5f) * static_cast<float>(src_w) / static_cast<float>(w)) - 0.5f;
                            const float cb = sample_chroma(sx, sy, true);
                            const float cr = sample_chroma(sx, sy, false);
                            const float r = Clamp01(yv + 1.402f * (cr - 0.5f));
                            const float g = Clamp01(yv - 0.344136f * (cb - 0.5f) - 0.714136f * (cr - 0.5f));
                            const float b = Clamp01(yv + 1.772f * (cb - 0.5f));
                            PIXEL_RGBA& d = drow[x];
                            d.r = FastClampToU8(r);
                            d.g = FastClampToU8(g);
                            d.b = FastClampToU8(b);
                            d.a = 255;
                        }
                    }
                };
                RowThreadPool::Instance().ParallelForRows(h, ChooseMinRowsPerTask(w, h), work);
            } else {
                auto work = [&](int y0, int y1) {
                    for (int y = y0; y < y1; ++y) {
                        const size_t row_base = static_cast<size_t>(y) * static_cast<size_t>(w);
                        PIXEL_RGBA* __restrict drow = out_pixels.data() + row_base;
                        const float* __restrict yrow = src + row_base;
                        for (int x = 0; x < w; ++x) {
                            const uint8_t v = FastClampToU8(yrow[x]);
                            PIXEL_RGBA& d = drow[x];
                            d.r = v;
                            d.g = v;
                            d.b = v;
                            d.a = 255;
                        }
                    }
                };
                RowThreadPool::Instance().ParallelForRows(h, ChooseMinRowsPerTask(w, h), work);
            }
        }
    }

    static void UnpackFromFloat16Buffer(const uint16_t* src, int w, int h, int channels, std::vector<PIXEL_RGBA>& out_pixels,
                                       const PIXEL_RGBA* src_pixels = nullptr, int src_w = 0, int src_h = 0) {
        const size_t plane = static_cast<size_t>(w) * static_cast<size_t>(h);
        const size_t plane2 = plane * 2;
        const size_t plane3 = plane * 3;
        if (out_pixels.size() != plane) out_pixels.resize(plane);
        const auto& lut = GetScalarLUTs();
        if (channels == 4) {
            auto work = [&](int y0, int y1) {
                for (int y = y0; y < y1; ++y) {
                    const size_t row_base = static_cast<size_t>(y) * static_cast<size_t>(w);
                    PIXEL_RGBA* __restrict drow = out_pixels.data() + row_base;
                    const uint16_t* __restrict rrow = src + row_base;
                    const uint16_t* __restrict grow = src + plane + row_base;
                    const uint16_t* __restrict brow = src + plane2 + row_base;
                    const uint16_t* __restrict arow = src + plane3 + row_base;
                    for (int x = 0; x < w; ++x) {
                        PIXEL_RGBA& d = drow[x];
                        d.r = lut.half_to_u8[rrow[x]];
                        d.g = lut.half_to_u8[grow[x]];
                        d.b = lut.half_to_u8[brow[x]];
                        d.a = lut.half_to_u8[arow[x]];
                    }
                }
            };
            RowThreadPool::Instance().ParallelForRows(h, ChooseMinRowsPerTask(w, h), work);
        } else if (channels == 3) {
            auto work = [&](int y0, int y1) {
                for (int y = y0; y < y1; ++y) {
                    const size_t row_base = static_cast<size_t>(y) * static_cast<size_t>(w);
                    PIXEL_RGBA* __restrict drow = out_pixels.data() + row_base;
                    const uint16_t* __restrict rrow = src + row_base;
                    const uint16_t* __restrict grow = src + plane + row_base;
                    const uint16_t* __restrict brow = src + plane2 + row_base;
                    for (int x = 0; x < w; ++x) {
                        PIXEL_RGBA& d = drow[x];
                        d.r = lut.half_to_u8[rrow[x]];
                        d.g = lut.half_to_u8[grow[x]];
                        d.b = lut.half_to_u8[brow[x]];
                        d.a = 255;
                    }
                }
            };
            RowThreadPool::Instance().ParallelForRows(h, ChooseMinRowsPerTask(w, h), work);
        } else if (channels == 1) {
            if (src_pixels && src_w > 0 && src_h > 0) {
                auto sample_chroma = [&](float sx, float sy, bool cb) -> float {
                    sx = std::clamp(sx, 0.0f, static_cast<float>(src_w - 1));
                    sy = std::clamp(sy, 0.0f, static_cast<float>(src_h - 1));
                    const int x0 = static_cast<int>(sx);
                    const int y0 = static_cast<int>(sy);
                    const int x1 = std::min(x0 + 1, src_w - 1);
                    const int y1 = std::min(y0 + 1, src_h - 1);
                    const float fx = sx - static_cast<float>(x0);
                    const float fy = sy - static_cast<float>(y0);
                    const auto& p00 = src_pixels[static_cast<size_t>(y0) * static_cast<size_t>(src_w) + static_cast<size_t>(x0)];
                    const auto& p10 = src_pixels[static_cast<size_t>(y0) * static_cast<size_t>(src_w) + static_cast<size_t>(x1)];
                    const auto& p01 = src_pixels[static_cast<size_t>(y1) * static_cast<size_t>(src_w) + static_cast<size_t>(x0)];
                    const auto& p11 = src_pixels[static_cast<size_t>(y1) * static_cast<size_t>(src_w) + static_cast<size_t>(x1)];
                    const float c00 = cb ? PixelCb01(p00) : PixelCr01(p00);
                    const float c10 = cb ? PixelCb01(p10) : PixelCr01(p10);
                    const float c01 = cb ? PixelCb01(p01) : PixelCr01(p01);
                    const float c11 = cb ? PixelCb01(p11) : PixelCr01(p11);
                    const float c0 = c00 + (c10 - c00) * fx;
                    const float c1 = c01 + (c11 - c01) * fx;
                    return c0 + (c1 - c0) * fy;
                };
                auto work = [&](int y0, int y1) {
                    for (int y = y0; y < y1; ++y) {
                        const size_t row_base = static_cast<size_t>(y) * static_cast<size_t>(w);
                        PIXEL_RGBA* __restrict drow = out_pixels.data() + row_base;
                        const uint16_t* __restrict yrow = src + row_base;
                        const float sy = ((static_cast<float>(y) + 0.5f) * static_cast<float>(src_h) / static_cast<float>(h)) - 0.5f;
                        for (int x = 0; x < w; ++x) {
                            const float yv = static_cast<float>(lut.half_to_u8[yrow[x]]) * (1.0f / 255.0f);
                            const float sx = ((static_cast<float>(x) + 0.5f) * static_cast<float>(src_w) / static_cast<float>(w)) - 0.5f;
                            const float cb = sample_chroma(sx, sy, true);
                            const float cr = sample_chroma(sx, sy, false);
                            const float r = Clamp01(yv + 1.402f * (cr - 0.5f));
                            const float g = Clamp01(yv - 0.344136f * (cb - 0.5f) - 0.714136f * (cr - 0.5f));
                            const float b = Clamp01(yv + 1.772f * (cb - 0.5f));
                            PIXEL_RGBA& d = drow[x];
                            d.r = FastClampToU8(r);
                            d.g = FastClampToU8(g);
                            d.b = FastClampToU8(b);
                            d.a = 255;
                        }
                    }
                };
                RowThreadPool::Instance().ParallelForRows(h, ChooseMinRowsPerTask(w, h), work);
            } else {
                auto work = [&](int y0, int y1) {
                    for (int y = y0; y < y1; ++y) {
                        const size_t row_base = static_cast<size_t>(y) * static_cast<size_t>(w);
                        PIXEL_RGBA* __restrict drow = out_pixels.data() + row_base;
                        const uint16_t* __restrict yrow = src + row_base;
                        for (int x = 0; x < w; ++x) {
                            const uint8_t v = lut.half_to_u8[yrow[x]];
                            PIXEL_RGBA& d = drow[x];
                            d.r = v;
                            d.g = v;
                            d.b = v;
                            d.a = 255;
                        }
                    }
                };
                RowThreadPool::Instance().ParallelForRows(h, ChooseMinRowsPerTask(w, h), work);
            }
        }
    }
    static bool FillPackedTensorFloatBuffer(const ml::TensorFloat& tensor, const std::vector<float>& packed, std::wstring& error) {
        try {
            auto ref = tensor.CreateReference();
            auto access = ref.as<IMemoryBufferByteAccess>();
            BYTE* bytes = nullptr;
            UINT32 capacity = 0;
            if (FAILED(access->GetBuffer(&bytes, &capacity)) || !bytes) {
                error = L"TensorFloat バッファ取得失敗";
                return false;
            }
            const size_t needed = packed.size() * sizeof(float);
            if (capacity < needed) {
                error = L"TensorFloat バッファサイズ不足";
                return false;
            }
            std::memcpy(bytes, packed.data(), needed);
            return true;
        } catch (const winrt::hresult_error& e) {
            error = L"TensorFloat 書き込み失敗: " + HResultErrorToString(e);
            return false;
        }
    }

    static bool FillPackedTensorFloat16Buffer(const ml::TensorFloat16Bit& tensor, const std::vector<uint16_t>& packed, std::wstring& error) {
        try {
            auto ref = tensor.CreateReference();
            auto access = ref.as<IMemoryBufferByteAccess>();
            BYTE* bytes = nullptr;
            UINT32 capacity = 0;
            if (FAILED(access->GetBuffer(&bytes, &capacity)) || !bytes) {
                error = L"TensorFloat16Bit バッファ取得失敗";
                return false;
            }
            const size_t needed = packed.size() * sizeof(uint16_t);
            if (capacity < needed) {
                error = L"TensorFloat16Bit バッファサイズ不足";
                return false;
            }
            std::memcpy(bytes, packed.data(), needed);
            return true;
        } catch (const winrt::hresult_error& e) {
            error = L"TensorFloat16Bit 書き込み失敗: " + HResultErrorToString(e);
            return false;
        }
    }

    bool FillPackedInputTensorFloat32(const std::vector<float>& packed, std::wstring& error) {
        return FillPackedTensorFloatBuffer(input_tensor_f32_, packed, error);
    }

    bool FillPackedInputTensorFloat16(const std::vector<uint16_t>& packed, std::wstring& error) {
        return FillPackedTensorFloat16Buffer(input_tensor_f16_, packed, error);
    }

    static void PackSingleFrameFloat(const PIXEL_RGBA* src_pixels, int w, int h, int channels_per_frame, std::vector<float>& out) {
        const size_t plane = static_cast<size_t>(w) * static_cast<size_t>(h);
        const size_t needed = plane * static_cast<size_t>(channels_per_frame);
        if (out.size() != needed) out.resize(needed);
        const auto& lut = GetScalarLUTs();
        const size_t plane2 = plane * 2;
        const size_t plane3 = plane * 3;
        if (channels_per_frame == 4) {
            auto work = [&](int y0, int y1) {
                for (int y = y0; y < y1; ++y) {
                    const size_t row_base = static_cast<size_t>(y) * static_cast<size_t>(w);
                    const PIXEL_RGBA* __restrict srow = src_pixels + row_base;
                    float* __restrict rrow = out.data() + row_base;
                    float* __restrict grow = out.data() + plane + row_base;
                    float* __restrict brow = out.data() + plane2 + row_base;
                    float* __restrict arow = out.data() + plane3 + row_base;
                    for (int x = 0; x < w; ++x) {
                        const PIXEL_RGBA& s = srow[x];
                        rrow[x] = lut.f32[s.r];
                        grow[x] = lut.f32[s.g];
                        brow[x] = lut.f32[s.b];
                        arow[x] = lut.f32[s.a];
                    }
                }
            };
            RowThreadPool::Instance().ParallelForRows(h, ChooseMinRowsPerTask(w, h), work);
        } else if (channels_per_frame == 3) {
            auto work = [&](int y0, int y1) {
                for (int y = y0; y < y1; ++y) {
                    const size_t row_base = static_cast<size_t>(y) * static_cast<size_t>(w);
                    const PIXEL_RGBA* __restrict srow = src_pixels + row_base;
                    float* __restrict rrow = out.data() + row_base;
                    float* __restrict grow = out.data() + plane + row_base;
                    float* __restrict brow = out.data() + plane2 + row_base;
                    for (int x = 0; x < w; ++x) {
                        const PIXEL_RGBA& s = srow[x];
                        rrow[x] = lut.f32[s.r];
                        grow[x] = lut.f32[s.g];
                        brow[x] = lut.f32[s.b];
                    }
                }
            };
            RowThreadPool::Instance().ParallelForRows(h, ChooseMinRowsPerTask(w, h), work);
        } else if (channels_per_frame == 1) {
            auto work = [&](int y0, int y1) {
                for (int y = y0; y < y1; ++y) {
                    const size_t row_base = static_cast<size_t>(y) * static_cast<size_t>(w);
                    const PIXEL_RGBA* __restrict srow = src_pixels + row_base;
                    float* __restrict yrow = out.data() + row_base;
                    for (int x = 0; x < w; ++x) {
                        const PIXEL_RGBA& s = srow[x];
                        const int gray = (77 * int(s.r) + 150 * int(s.g) + 29 * int(s.b) + 128) >> 8;
                        yrow[x] = lut.f32[gray];
                    }
                }
            };
            RowThreadPool::Instance().ParallelForRows(h, ChooseMinRowsPerTask(w, h), work);
        }
    }

    static void PackSingleFrameHalf(const PIXEL_RGBA* src_pixels, int w, int h, int channels_per_frame, std::vector<uint16_t>& out) {
        const size_t plane = static_cast<size_t>(w) * static_cast<size_t>(h);
        const size_t needed = plane * static_cast<size_t>(channels_per_frame);
        if (out.size() != needed) out.resize(needed);
        const auto& lut = GetScalarLUTs();
        const size_t plane2 = plane * 2;
        const size_t plane3 = plane * 3;
        if (channels_per_frame == 4) {
            auto work = [&](int y0, int y1) {
                for (int y = y0; y < y1; ++y) {
                    const size_t row_base = static_cast<size_t>(y) * static_cast<size_t>(w);
                    const PIXEL_RGBA* __restrict srow = src_pixels + row_base;
                    uint16_t* __restrict rrow = out.data() + row_base;
                    uint16_t* __restrict grow = out.data() + plane + row_base;
                    uint16_t* __restrict brow = out.data() + plane2 + row_base;
                    uint16_t* __restrict arow = out.data() + plane3 + row_base;
                    for (int x = 0; x < w; ++x) {
                        const PIXEL_RGBA& s = srow[x];
                        rrow[x] = lut.f16[s.r];
                        grow[x] = lut.f16[s.g];
                        brow[x] = lut.f16[s.b];
                        arow[x] = lut.f16[s.a];
                    }
                }
            };
            RowThreadPool::Instance().ParallelForRows(h, ChooseMinRowsPerTask(w, h), work);
        } else if (channels_per_frame == 3) {
            auto work = [&](int y0, int y1) {
                for (int y = y0; y < y1; ++y) {
                    const size_t row_base = static_cast<size_t>(y) * static_cast<size_t>(w);
                    const PIXEL_RGBA* __restrict srow = src_pixels + row_base;
                    uint16_t* __restrict rrow = out.data() + row_base;
                    uint16_t* __restrict grow = out.data() + plane + row_base;
                    uint16_t* __restrict brow = out.data() + plane2 + row_base;
                    for (int x = 0; x < w; ++x) {
                        const PIXEL_RGBA& s = srow[x];
                        rrow[x] = lut.f16[s.r];
                        grow[x] = lut.f16[s.g];
                        brow[x] = lut.f16[s.b];
                    }
                }
            };
            RowThreadPool::Instance().ParallelForRows(h, ChooseMinRowsPerTask(w, h), work);
        } else if (channels_per_frame == 1) {
            auto work = [&](int y0, int y1) {
                for (int y = y0; y < y1; ++y) {
                    const size_t row_base = static_cast<size_t>(y) * static_cast<size_t>(w);
                    const PIXEL_RGBA* __restrict srow = src_pixels + row_base;
                    uint16_t* __restrict yrow = out.data() + row_base;
                    for (int x = 0; x < w; ++x) {
                        const PIXEL_RGBA& s = srow[x];
                        const int gray = (77 * int(s.r) + 150 * int(s.g) + 29 * int(s.b) + 128) >> 8;
                        yrow[x] = lut.f16[gray];
                    }
                }
            };
            RowThreadPool::Instance().ParallelForRows(h, ChooseMinRowsPerTask(w, h), work);
        }
    }

    void PrepareTemporalWindow(const PIXEL_RGBA* src_pixels, int w, int h, int frame_count,
                              int object_frame, int64_t object_id, int64_t effect_id) {
        const bool size_changed = (w != last_window_width_ || h != last_window_height_);
        const bool object_changed = (object_id != last_object_id_ || effect_id != last_effect_id_);
        const bool first_frame = !has_last_frame_;
        const bool repeated_frame = has_last_frame_ && object_frame == last_frame_number_;
        const bool sequential_frame = has_last_frame_ && object_frame == (last_frame_number_ + 1);
        const bool discontinuous_frame = !(first_frame || repeated_frame || sequential_frame);

        const int channels_per_frame = static_cast<int>(model_spec_.channels_per_frame);
        const size_t plane = static_cast<size_t>(w) * static_cast<size_t>(h);
        const size_t frame_elems = plane * static_cast<size_t>(channels_per_frame);
        const size_t total_elems = frame_elems * static_cast<size_t>(std::max(frame_count, 1));

        if (size_changed || object_changed || discontinuous_frame) {
            packed_input_f32_.clear();
            packed_input_f16_.clear();
            last_output_pixels_.clear();
            last_output_width_ = 0;
            last_output_height_ = 0;
            last_window_width_ = w;
            last_window_height_ = h;
            DebugOutHotPath((std::wstring(L"vsr window init frame=") + std::to_wstring(object_frame) + L" size=" + std::to_wstring(frame_count)).c_str());
        }

        if (model_spec_.input_kind == ml::TensorKind::Float16) {
            PackSingleFrameHalf(src_pixels, w, h, channels_per_frame, frame_scratch_f16_);
            auto& current = frame_scratch_f16_;

            if (repeated_frame && packed_input_f16_.size() == total_elems) {
                DebugOutHotPath((std::wstring(L"vsr window frame=") + std::to_wstring(object_frame) + L" repeated=1 sequential=0 count=" + std::to_wstring(frame_count)).c_str());
                return;
            }

            if (packed_input_f16_.size() != total_elems) packed_input_f16_.resize(total_elems);

            if (first_frame || object_changed || discontinuous_frame || size_changed) {
                for (int i = 0; i < frame_count; ++i) {
                    std::memcpy(packed_input_f16_.data() + static_cast<size_t>(i) * frame_elems,
                                current.data(), frame_elems * sizeof(uint16_t));
                }
            } else if (sequential_frame) {
                if (frame_count > 1) {
                    std::memmove(packed_input_f16_.data(),
                                 packed_input_f16_.data() + frame_elems,
                                 (static_cast<size_t>(frame_count - 1) * frame_elems) * sizeof(uint16_t));
                }
                std::memcpy(packed_input_f16_.data() + static_cast<size_t>(frame_count - 1) * frame_elems,
                            current.data(), frame_elems * sizeof(uint16_t));
            } else {
                for (int i = 0; i < frame_count; ++i) {
                    std::memcpy(packed_input_f16_.data() + static_cast<size_t>(i) * frame_elems,
                                current.data(), frame_elems * sizeof(uint16_t));
                }
            }
        } else {
            PackSingleFrameFloat(src_pixels, w, h, channels_per_frame, frame_scratch_f32_);
            auto& current = frame_scratch_f32_;

            if (repeated_frame && packed_input_f32_.size() == total_elems) {
                DebugOutHotPath((std::wstring(L"vsr window frame=") + std::to_wstring(object_frame) + L" repeated=1 sequential=0 count=" + std::to_wstring(frame_count)).c_str());
                return;
            }

            if (packed_input_f32_.size() != total_elems) packed_input_f32_.resize(total_elems);

            if (first_frame || object_changed || discontinuous_frame || size_changed) {
                for (int i = 0; i < frame_count; ++i) {
                    std::memcpy(packed_input_f32_.data() + static_cast<size_t>(i) * frame_elems,
                                current.data(), frame_elems * sizeof(float));
                }
            } else if (sequential_frame) {
                if (frame_count > 1) {
                    std::memmove(packed_input_f32_.data(),
                                 packed_input_f32_.data() + frame_elems,
                                 (static_cast<size_t>(frame_count - 1) * frame_elems) * sizeof(float));
                }
                std::memcpy(packed_input_f32_.data() + static_cast<size_t>(frame_count - 1) * frame_elems,
                            current.data(), frame_elems * sizeof(float));
            } else {
                for (int i = 0; i < frame_count; ++i) {
                    std::memcpy(packed_input_f32_.data() + static_cast<size_t>(i) * frame_elems,
                                current.data(), frame_elems * sizeof(float));
                }
            }
        }

        {
            std::wstringstream ss;
            ss << L"vsr window frame=" << object_frame
               << L" repeated=" << (repeated_frame ? 1 : 0)
               << L" sequential=" << (sequential_frame ? 1 : 0)
               << L" count=" << frame_count;
            DebugOutHotPath(ss.str().c_str());
        }

        has_last_frame_ = true;
        last_frame_number_ = object_frame;
        last_object_id_ = object_id;
        last_effect_id_ = effect_id;
    }

    void PackTemporalWindowFloat(int w, int h, int channels_per_frame, std::vector<float>& out) const {
        const size_t plane = static_cast<size_t>(w) * static_cast<size_t>(h);
        const size_t frame_elems = plane * static_cast<size_t>(channels_per_frame);
        const size_t expected = frame_elems * static_cast<size_t>(std::max<int64_t>(1, model_spec_.frame_count));
        if (&out != &packed_input_f32_) {
            out = packed_input_f32_;
        }
        if (out.size() != expected) {
            out.resize(expected);
        }
    }

    void PackTemporalWindowHalf(int w, int h, int channels_per_frame, std::vector<uint16_t>& out) const {
        const size_t plane = static_cast<size_t>(w) * static_cast<size_t>(h);
        const size_t frame_elems = plane * static_cast<size_t>(channels_per_frame);
        const size_t expected = frame_elems * static_cast<size_t>(std::max<int64_t>(1, model_spec_.frame_count));
        if (&out != &packed_input_f16_) {
            out = packed_input_f16_;
        }
        if (out.size() != expected) {
            out.resize(expected);
        }
    }

    static bool UnpackOutputTensorFloat32(const ml::TensorFloat& tensor, int w, int h, int channels,
                                          std::vector<PIXEL_RGBA>& out_pixels, std::wstring& error,
                                          const PIXEL_RGBA* src_pixels = nullptr, int src_w = 0, int src_h = 0) {
        try {
            auto ref = tensor.CreateReference();
            auto access = ref.as<IMemoryBufferByteAccess>();
            BYTE* bytes = nullptr;
            UINT32 capacity = 0;
            if (FAILED(access->GetBuffer(&bytes, &capacity)) || !bytes) {
                error = L"TensorFloat 出力バッファ取得失敗";
                return false;
            }
            const size_t needed = static_cast<size_t>(w) * static_cast<size_t>(h) * static_cast<size_t>(channels) * sizeof(float);
            if (capacity < needed) {
                error = L"TensorFloat 出力バッファサイズ不足";
                return false;
            }
            UnpackFromFloatBuffer(reinterpret_cast<const float*>(bytes), w, h, channels, out_pixels, src_pixels, src_w, src_h);
            return true;
        } catch (const winrt::hresult_error& e) {
            error = L"TensorFloat 出力読み出し失敗: " + HResultErrorToString(e);
            return false;
        }
    }

    static bool UnpackOutputTensorFloat16(const ml::TensorFloat16Bit& tensor, int w, int h, int channels,
                                          std::vector<PIXEL_RGBA>& out_pixels, std::wstring& error,
                                          const PIXEL_RGBA* src_pixels = nullptr, int src_w = 0, int src_h = 0) {
        try {
            auto ref = tensor.CreateReference();
            auto access = ref.as<IMemoryBufferByteAccess>();
            BYTE* bytes = nullptr;
            UINT32 capacity = 0;
            if (FAILED(access->GetBuffer(&bytes, &capacity)) || !bytes) {
                error = L"TensorFloat16Bit 出力バッファ取得失敗";
                return false;
            }
            const size_t needed = static_cast<size_t>(w) * static_cast<size_t>(h) * static_cast<size_t>(channels) * sizeof(uint16_t);
            if (capacity < needed) {
                error = L"TensorFloat16Bit 出力バッファサイズ不足";
                return false;
            }
            UnpackFromFloat16Buffer(reinterpret_cast<const uint16_t*>(bytes), w, h, channels, out_pixels, src_pixels, src_w, src_h);
            return true;
        } catch (const winrt::hresult_error& e) {
            error = L"TensorFloat16Bit 出力読み出し失敗: " + HResultErrorToString(e);
            return false;
        }
    }

private:
    void InvalidateSISRExecutionState() {
        bound_input_shape_.clear();
        input_tensor_f32_ = nullptr;
        input_tensor_f16_ = nullptr;
        bound_output_shape_.clear();
        output_tensor_f32_ = nullptr;
        output_tensor_f16_ = nullptr;
        output_binding_ready_ = false;
        cached_binding_ = nullptr;
        binding_ready_ = false;
        defer_output_binding_for_eval_ = false;
    }

    ml::LearningModel model_{ nullptr };
    ml::LearningModelDevice device_{ nullptr };
    ml::LearningModelSession session_{ nullptr };
    ModelSpec model_spec_{};
    bool loaded_ = false;
    std::wstring backend_name_;
    std::wstring selected_device_name_;
    std::wstring model_path_;
    std::vector<uint8_t> absorbed_model_bytes_;
    std::vector<int64_t> bound_input_shape_;
    ml::TensorFloat input_tensor_f32_{ nullptr };
    ml::TensorFloat16Bit input_tensor_f16_{ nullptr };
    std::vector<int64_t> bound_output_shape_;
    ml::TensorFloat output_tensor_f32_{ nullptr };
    ml::TensorFloat16Bit output_tensor_f16_{ nullptr };
    bool output_binding_ready_ = false;
    bool defer_output_binding_for_eval_ = false;
    ml::LearningModelBinding cached_binding_{ nullptr };
    bool binding_ready_ = false;
    std::vector<float> packed_input_f32_;
    std::vector<uint16_t> packed_input_f16_;
    std::vector<float> frame_scratch_f32_;
    std::vector<uint16_t> frame_scratch_f16_;
    std::vector<PIXEL_RGBA> last_output_pixels_;
    int last_output_width_ = 0;
    int last_output_height_ = 0;
    int last_window_width_ = 0;
    int last_window_height_ = 0;
    int last_sisr_input_width_ = 0;
    int last_sisr_input_height_ = 0;
    bool has_last_frame_ = false;
    int last_frame_number_ = std::numeric_limits<int>::min();
    int64_t last_object_id_ = 0;
    int64_t last_effect_id_ = 0;
};
static std::mutex g_mutex;
static std::unique_ptr<WinMLEngine> g_engine;
static std::wstring g_last_error;
static std::wstring g_loaded_model_path;
static int g_last_backend_mode = 0;
static std::wstring g_last_device_name;

static void ReleaseEngineResources() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_engine) {
        g_engine->Reset();
        g_engine.reset();
    }
    g_loaded_model_path.clear();
    g_last_error.clear();
    g_last_device_name.clear();
}

static bool ValidateFrameSize(int width, int height, std::wstring& error) {
    if (width <= 0 || height <= 0) {
        error = L"不正サイズ";
        return false;
    }

    const uint64_t w = static_cast<uint64_t>(width);
    const uint64_t h = static_cast<uint64_t>(height);
    const uint64_t pixels = w * h;
    const uint64_t max_pixels = 7680ull * 4320ull;
    if (pixels == 0 || pixels > max_pixels) {
        error = L"入力サイズが大きすぎます";
        return false;
    }

    if (pixels > (static_cast<uint64_t>(std::numeric_limits<size_t>::max()) / sizeof(PIXEL_RGBA)) ||
        pixels > (static_cast<uint64_t>(std::numeric_limits<size_t>::max()) / 4ull)) {
        error = L"入力バッファサイズが不正です";
        return false;
    }

    return true;
}

static bool EnsureEngineLoaded() {
    std::lock_guard<std::mutex> lock(g_mutex);

    const std::wstring model_path = GetConfiguredModelPath();
    const int backend_mode = g_backend.value;
    const bool need_reload = !g_engine || !g_engine->IsLoaded() || g_loaded_model_path != model_path || g_last_backend_mode != backend_mode;
    if (!need_reload) return true;

    if (model_path.empty()) {
        g_last_error = L"ONNXファイルが未選択です";
        g_engine.reset();
        return false;
    }

    if (!FileExists(model_path)) {
        g_last_error = L"モデルが見つかりません";
        g_engine.reset();
        return false;
    }

    auto engine = std::make_unique<WinMLEngine>();
    std::wstring err;
    if (!engine->Load(model_path, backend_mode, err)) {
        g_last_error = err;
        g_engine.reset();
        return false;
    }

    g_last_error = err;
    g_loaded_model_path = model_path;
    g_last_backend_mode = backend_mode;
    g_engine = std::move(engine);
    g_last_device_name = g_engine ? g_engine->SelectedDeviceName() : L"";
    if (g_engine) {
        const auto& spec = g_engine->Spec();
        std::wstringstream ss;
        ss << L"engine loaded mode=" << ModelModeName(spec.mode)
           << L" in_c=" << spec.in_c
           << L" out_c=" << spec.out_c
           << L" frames=" << spec.frame_count
           << L" cpf=" << spec.channels_per_frame;
        if (!g_last_device_name.empty()) ss << L" device=" << g_last_device_name;
        DebugOut(ss.str().c_str());
    }
    return true;
}

static bool func_proc_video_impl(FILTER_PROC_VIDEO* video) {
    if (!video || !video->scene || !video->object || !video->get_image_data || !video->set_image_data) return false;
    if (!g_enable.value) return true;

    const int width = video->object->width;
    const int height = video->object->height;
    if (width <= 0 || height <= 0) return true;

    {
        std::wstringstream ss;
        ss << L"proc frame=" << video->object->frame
           << L" obj=" << video->object->id
           << L" eff=" << video->object->effect_id
           << L" size=" << width << L"x" << height;
        DebugOutHotPath(ss.str().c_str());
    }

    std::wstring size_error;
    if (!ValidateFrameSize(width, height, size_error)) {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_last_error = size_error;
        DebugOut(g_last_error.c_str());
        return true;
    }

    static thread_local std::vector<PIXEL_RGBA> src_pixels;
    const size_t frame_pixels = (size_t)width * (size_t)height;
    if (src_pixels.size() != frame_pixels) src_pixels.resize(frame_pixels);
    video->get_image_data(src_pixels.data());

    if (!EnsureEngineLoaded()) {
        std::lock_guard<std::mutex> lock(g_mutex);
        DebugOut(g_last_error.c_str());
        return true;
    }

    static thread_local std::vector<PIXEL_RGBA> dst_pixels;
    int out_w = 0, out_h = 0;
    std::wstring err;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!g_engine || !g_engine->IsLoaded()) {
            g_last_error = L"モデル未読込";
            DebugOut(g_last_error.c_str());
            return true;
        }
        if (!g_engine->RunPixels(src_pixels.data(), width, height, video->object->frame, video->object->id, video->object->effect_id, dst_pixels, out_w, out_h, err)) {
            g_last_error = err.empty() ? L"推論に失敗しました" : err;
            DebugOut(g_last_error.c_str());
            return true;
        }
    }

    if (out_w <= 0 || out_h <= 0 || dst_pixels.size() != (size_t)out_w * (size_t)out_h) {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_last_error = L"出力サイズが不正です";
        DebugOut(g_last_error.c_str());
        return true;
    }

    if (video->object->is_filter_object() && (out_w != width || out_h != height)) {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_last_error = L"この出力サイズはフィルタオブジェクトでは使用できません";
        DebugOut(g_last_error.c_str());
        return true;
    }

    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_last_error.clear();
    }
    video->set_image_data(dst_pixels.data(), out_w, out_h);
    return true;
}

static bool func_proc_video(FILTER_PROC_VIDEO* video) {
    try {
        return func_proc_video_impl(video);
    } catch (const std::bad_alloc&) {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_last_error = L"メモリ確保に失敗しました";
        DebugOut(g_last_error.c_str());
        return true;
    } catch (const std::exception&) {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_last_error = L"処理中に例外が発生しました";
        DebugOut(g_last_error.c_str());
        return true;
    } catch (...) {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_last_error = L"処理中に不明な例外が発生しました";
        DebugOut(g_last_error.c_str());
        return true;
    }
}

#define PLUGIN_NAME         L"AI S-R ONNX"
#define PLUGIN_VERSION      L"1.1"
#define PLUGIN_CATEGORY     L"加工"
#define PLUGIN_INFORMATION  PLUGIN_NAME L" " PLUGIN_VERSION

static FILTER_PLUGIN_TABLE g_filter_plugin_table = {
    FILTER_PLUGIN_TABLE::FLAG_VIDEO,
    PLUGIN_NAME,
    PLUGIN_CATEGORY,
    PLUGIN_INFORMATION,
    g_items,
    func_proc_video,
    nullptr
};

EXTERN_C __declspec(dllexport) bool InitializePlugin(DWORD) {
    DebugOut(L"AI S-R ONNX InitializePlugin");
    UpdateBackendLabels();
    DebugOut(g_backend_label_auto);
    DebugOut(g_backend_label_gpu0);
    DebugOut(g_backend_label_gpu1);
    return true;
}

EXTERN_C __declspec(dllexport) void UninitializePlugin() {
    DebugOut(L"AI S-R ONNX UninitializePlugin");
    ReleaseEngineResources();
}

EXTERN_C __declspec(dllexport) FILTER_PLUGIN_TABLE* GetFilterPluginTable(void) {
    return &g_filter_plugin_table;
}
