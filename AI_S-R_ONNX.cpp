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
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <memory>
#include <limits>
#include <thread>
#include <condition_variable>
#include <functional>
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

namespace ml = winrt::Windows::AI::MachineLearning;
namespace wfc = winrt::Windows::Foundation::Collections;
namespace wg = winrt::Windows::Graphics;
namespace wgdx11 = winrt::Windows::Graphics::DirectX::Direct3D11;
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

// Debug logging is compiled out in release builds.
static void DebugOut(const wchar_t* text) {
#ifdef _DEBUG
    OutputDebugStringW(text);
    OutputDebugStringW(L"\n");
#else
    (void)text;
#endif
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
struct ModelSpec {
    winrt::hstring input_name;
    winrt::hstring output_name;
    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;
    int64_t in_c = 0;
    int64_t out_c = 0;
    ml::TensorKind input_kind = ml::TensorKind::Undefined;
    ml::TensorKind output_kind = ml::TensorKind::Undefined;
    bool valid = false;

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

            model_ = ml::LearningModel::LoadFromFilePath(model_path);
            model_path_ = model_path;
            model_spec_ = InspectModel(model_);
            if (!model_spec_.valid) {
                error = L"対応外モデル: float32/float16 / 1入力1出力 / 4次元NCHWのみ対応";
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
        } catch (const winrt::hresult_error&) {
            error = L"モデルの読み込みに失敗しました";
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
        bound_input_shape_.clear();
        input_tensor_f32_ = nullptr;
        input_tensor_f16_ = nullptr;
        cached_binding_ = nullptr;
        binding_ready_ = false;
    }

    bool IsLoaded() const { return loaded_ && session_ != nullptr; }
    const std::wstring& BackendName() const { return backend_name_; }
    const std::wstring& SelectedDeviceName() const { return selected_device_name_; }
    const std::wstring& ModelPath() const { return model_path_; }

    bool RunPixels(const PIXEL_RGBA* src_pixels, int width, int height, std::vector<PIXEL_RGBA>& dst_pixels,
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

        const int in_c = static_cast<int>(model_spec_.in_c);
        if (!(in_c == 3 || in_c == 4)) {
            error = L"入力チャネル数が3/4ではありません";
            return false;
        }

        auto input_shape = ResolveInputShape(width, height);
        if (input_shape.size() != 4) {
            error = L"入力shape解決失敗";
            return false;
        }

        if (!EnsureInputTensor(input_shape, error)) {
            return false;
        }

        if (model_spec_.input_kind == ml::TensorKind::Float16) {
            if (!FillInputTensorFloat16(src_pixels, width, height, in_c, error)) {
                return false;
            }
        } else {
            if (!FillInputTensorFloat32(src_pixels, width, height, in_c, error)) {
                return false;
            }
        }
        if (!EnsureInputBinding(error)) {
            return false;
        }

        auto result = session_.Evaluate(cached_binding_, L"AI S-R ONNX");
        auto outputs = result.Outputs();
        winrt::Windows::Foundation::IInspectable boxed{ nullptr };
        try {
            boxed = outputs.Lookup(model_spec_.output_name);
        } catch (const winrt::hresult_error&) {
            error = L"出力名が見つかりません";
            return false;
        }
        if (model_spec_.output_kind == ml::TensorKind::Float16) {
            auto out_tensor = boxed.try_as<ml::TensorFloat16Bit>();
            if (!out_tensor) {
                error = L"出力テンソルがTensorFloat16Bitではありません";
                return false;
            }
            auto out_shape = ToStdVector(out_tensor.Shape());
            if (!ValidateOutputShape(out_shape, out_width, out_height, error)) {
                return false;
            }
            if (!UnpackOutputTensorFloat16(out_tensor, out_width, out_height, static_cast<int>(out_shape[1]), dst_pixels, error)) {
                return false;
            }
        } else {
            auto out_tensor = boxed.try_as<ml::TensorFloat>();
            if (!out_tensor) {
                error = L"出力テンソルがTensorFloatではありません";
                return false;
            }
            auto out_shape = ToStdVector(out_tensor.Shape());
            if (!ValidateOutputShape(out_shape, out_width, out_height, error)) {
                return false;
            }
            if (!UnpackOutputTensorFloat32(out_tensor, out_width, out_height, static_cast<int>(out_shape[1]), dst_pixels, error)) {
                return false;
            }
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
        if (!(spec.in_c == 3 || spec.in_c == 4 || spec.in_c <= 0)) return spec;
        if (!(spec.out_c == 3 || spec.out_c == 4 || spec.out_c <= 0)) return spec;

        if (spec.in_c <= 0) spec.in_c = 3;
        if (spec.out_c <= 0) spec.out_c = 3;
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

    // Cache input binding only. Output binding stays dynamic for stability.
    bool EnsureInputBinding(std::wstring& error) {
        try {
            if (!cached_binding_) {
                cached_binding_ = ml::LearningModelBinding(session_);
                binding_ready_ = false;
            }
            if (!binding_ready_) {
                if (model_spec_.input_kind == ml::TensorKind::Float16) {
                    cached_binding_.Bind(model_spec_.input_name, input_tensor_f16_);
                } else {
                    cached_binding_.Bind(model_spec_.input_name, input_tensor_f32_);
                }
                binding_ready_ = true;
            }
            return true;
        } catch (const winrt::hresult_error& e) {
            error = L"入力バインド作成失敗: " + HResultErrorToString(e);
            cached_binding_ = nullptr;
            binding_ready_ = false;
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
        if (out_n != 1 || !(out_c == 3 || out_c == 4) || out_w <= 0 || out_h <= 0) {
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

    // Pack planar NCHW input into the already-allocated WinML tensor.
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
            } else {
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
            } else {
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

    static void UnpackFromFloatBuffer(const float* src, int w, int h, int channels, std::vector<PIXEL_RGBA>& out_pixels) {
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
        } else {
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
        }
    }

    static void UnpackFromFloat16Buffer(const uint16_t* src, int w, int h, int channels, std::vector<PIXEL_RGBA>& out_pixels) {
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
        } else {
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
        }
    }

    static bool UnpackOutputTensorFloat32(const ml::TensorFloat& tensor, int w, int h, int channels,
                                          std::vector<PIXEL_RGBA>& out_pixels, std::wstring& error) {
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
            UnpackFromFloatBuffer(reinterpret_cast<const float*>(bytes), w, h, channels, out_pixels);
            return true;
        } catch (const winrt::hresult_error& e) {
            error = L"TensorFloat 出力読み出し失敗: " + HResultErrorToString(e);
            return false;
        }
    }

    static bool UnpackOutputTensorFloat16(const ml::TensorFloat16Bit& tensor, int w, int h, int channels,
                                          std::vector<PIXEL_RGBA>& out_pixels, std::wstring& error) {
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
            UnpackFromFloat16Buffer(reinterpret_cast<const uint16_t*>(bytes), w, h, channels, out_pixels);
            return true;
        } catch (const winrt::hresult_error& e) {
            error = L"TensorFloat16Bit 出力読み出し失敗: " + HResultErrorToString(e);
            return false;
        }
    }

private:
    ml::LearningModel model_{ nullptr };
    ml::LearningModelDevice device_{ nullptr };
    ml::LearningModelSession session_{ nullptr };
    ModelSpec model_spec_{};
    bool loaded_ = false;
    std::wstring backend_name_;
    std::wstring selected_device_name_;
    std::wstring model_path_;
    std::vector<int64_t> bound_input_shape_;
    ml::TensorFloat input_tensor_f32_{ nullptr };
    ml::TensorFloat16Bit input_tensor_f16_{ nullptr };
    ml::LearningModelBinding cached_binding_{ nullptr };
    bool binding_ready_ = false;
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
    return true;
}

static bool func_proc_video_impl(FILTER_PROC_VIDEO* video) {
    if (!video || !video->scene || !video->object || !video->get_image_data || !video->set_image_data) return false;
    if (!g_enable.value) return true;

    const int width = video->object->width;
    const int height = video->object->height;
    if (width <= 0 || height <= 0) return true;

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
        if (!g_engine->RunPixels(src_pixels.data(), width, height, dst_pixels, out_w, out_h, err)) {
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

#define PLUGIN_NAME      L"AI S-R ONNX"
#define PLUGIN_VERSION   L"1.0"
#define PLUGIN_TITLE     PLUGIN_NAME L" " PLUGIN_VERSION
#define PLUGIN_MENU_NAME PLUGIN_NAME
#define PLUGIN_INFO_NAME PLUGIN_TITLE

static FILTER_PLUGIN_TABLE g_filter_plugin_table = {
    FILTER_PLUGIN_TABLE::FLAG_VIDEO,
    PLUGIN_MENU_NAME,
    PLUGIN_INFO_NAME,
    PLUGIN_INFO_NAME,
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
