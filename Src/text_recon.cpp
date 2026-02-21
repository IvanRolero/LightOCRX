/* LightOCRX Lightweight OCR software
 *
 * Copyright (C) 2026 Ivan Rolero
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "text_recon.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <locale>
#include <codecvt>
#include <cmath>

#include "include/onnxruntime_cxx_api.h"

#ifdef _WIN32
static std::wstring to_wstring_utf8(const std::string& s) {
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> conv;
    return conv.from_bytes(s);
}
#endif

// Helper function to compute dynamic width
int TextRecognizer::compute_dynamic_width(int h, int w, int base_h, int min_w, int max_w) {
    float aspect_ratio = static_cast<float>(w) / std::max(h, 1);
    int dynamic_width = static_cast<int>(base_h * aspect_ratio);
    
    // Ensure width is divisible by 8 for some models (optional)
    dynamic_width = ((dynamic_width + 7) / 8) * 8;
    
    // Clamp to reasonable limits
    dynamic_width = std::max(min_w, std::min(dynamic_width, max_w));
    
    return dynamic_width;
}

TextRecognizer::TextRecognizer(const std::string& rec_model_path, const ReconOptions& opt)
    : opt_(opt),
      env_(ORT_LOGGING_LEVEL_WARNING, "ocr") {
    load_dict_or_default(opt_.dict_path);

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

#ifdef _WIN32
    session_ = Ort::Session(env_, to_wstring_utf8(rec_model_path).c_str(), session_options);
#else
    session_ = Ort::Session(env_, rec_model_path.c_str(), session_options);
#endif

    Ort::AllocatorWithDefaultOptions allocator;

    size_t num_input_nodes = session_.GetInputCount();
    for (size_t i = 0; i < num_input_nodes; i++) {
        auto name = session_.GetInputNameAllocated(i, allocator);
        input_names_str_.push_back(name.get());
        input_names_.push_back(input_names_str_.back().c_str());
    }

    size_t num_output_nodes = session_.GetOutputCount();
    for (size_t i = 0; i < num_output_nodes; i++) {
        auto name = session_.GetOutputNameAllocated(i, allocator);
        output_names_str_.push_back(name.get());
        output_names_.push_back(output_names_str_.back().c_str());
    }

    if (input_names_.empty() || output_names_.empty()) {
        throw std::runtime_error("Failed to get input/output names from recognition model");
    }
}

void TextRecognizer::load_dict_or_default(const std::string& dict_path) {
    std::ifstream file(dict_path);
    if (!file.is_open()) {
        // default fallback
        char_dict_.clear();
        char_dict_.push_back("");
        for (int i = 1; i <= 10; i++) char_dict_.push_back(std::string(1, '0' + (i - 1)));
        for (int i = 11; i <= 36; i++) char_dict_.push_back(std::string(1, 'a' + (i - 11)));
        for (int i = 37; i <= 62; i++) char_dict_.push_back(std::string(1, 'A' + (i - 37)));
        return;
    }

    char_dict_.clear();
    std::string line;
    while (std::getline(file, line) && (int)char_dict_.size() < opt_.max_dict_size) {
        if (line.empty()) continue;

        if (line == " " || line == "\\s") char_dict_.push_back(" ");
        else if (line == "\\n") char_dict_.push_back("\n");
        else if (line == "\\t") char_dict_.push_back("\t");
        else char_dict_.push_back(line);
    }

    if (char_dict_.empty()) {
        throw std::runtime_error("Dictionary loaded but empty: " + dict_path);
    }
}

std::string TextRecognizer::decode_ctc(const float* output, int batch, int time_step, int class_num) {
    std::string result;

    for (int b = 0; b < batch; b++) {
        const float* batch_output = output + b * time_step * class_num;
        int prev_class_idx = -1;

        for (int t = 0; t < time_step; t++) {
            const float* timestep_output = batch_output + t * class_num;

            int max_idx = 0;
            float max_val = timestep_output[0];
            for (int c = 1; c < class_num; c++) {
                if (timestep_output[c] > max_val) {
                    max_val = timestep_output[c];
                    max_idx = c;
                }
            }

            if (max_idx != 0) {
                if (max_idx != prev_class_idx) {
                    int dict_idx = max_idx - 1;
                    if (dict_idx >= 0 && dict_idx < (int)char_dict_.size()) {
                        result += char_dict_[dict_idx];
                    }
                    prev_class_idx = max_idx;
                }
            } else {
                prev_class_idx = -1;
            }
        }
    }

    return result;
}

std::string TextRecognizer::recognize(const uint8_t* data, int w, int h, int c) {
    if (!data || w <= 0 || h <= 0) return "";

    // Force RGB input
    std::vector<uint8_t> rgb(w * h * 3);
    if (c >= 3) {
        for (int i = 0; i < w * h; i++) {
            rgb[i * 3 + 0] = data[i * c + 0];
            rgb[i * 3 + 1] = data[i * c + 1];
            rgb[i * 3 + 2] = data[i * c + 2];
        }
    } else {
        for (int i = 0; i < w * h; i++) {
            rgb[i * 3 + 0] = data[i];
            rgb[i * 3 + 1] = data[i];
            rgb[i * 3 + 2] = data[i];
        }
    }

    // Use dynamic width if target_w is 0 or negative, otherwise use fixed width
    int TH = opt_.target_h;
    int TW = opt_.target_w > 0 ? opt_.target_w : compute_dynamic_width(h, w, TH, 256, 4096);

    float scale_h = (float)TH / h;
    float scale_w = (float)TW / w;
    float scale = std::min(scale_h, scale_w);

    int new_w = std::max(1, (int)(w * scale));
    int new_h = std::max(1, (int)(h * scale));

    int start_x = (TW - new_w) / 2;
    int start_y = (TH - new_h) / 2;

    std::vector<uint8_t> resized(TH * TW * 3, 127);

    // simple nearest resize
    for (int y = 0; y < new_h && (start_y + y) < TH; y++) {
        int sy = (int)(y * (float)h / new_h);
        if (sy >= h) sy = h - 1;

        for (int x = 0; x < new_w && (start_x + x) < TW; x++) {
            int sx = (int)(x * (float)w / new_w);
            if (sx >= w) sx = w - 1;

            for (int ch = 0; ch < 3; ch++) {
                resized[((start_y + y) * TW + (start_x + x)) * 3 + ch] =
                    rgb[(sy * w + sx) * 3 + ch];
            }
        }
    }

    // Normalize to CHW float: v/127.5 - 1
    std::vector<float> input(3 * TH * TW);
    for (int y = 0; y < TH; y++) {
        for (int x = 0; x < TW; x++) {
            int base = (y * TW + x) * 3;
            for (int ch = 0; ch < 3; ch++) {
                float norm = resized[base + ch] / 127.5f - 1.0f;
                input[ch * (TH * TW) + y * TW + x] = norm;
            }
        }
    }

    std::vector<int64_t> input_shape = {1, 3, TH, TW};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input.data(),
        input.size(),
        input_shape.data(),
        input_shape.size());

    std::vector<Ort::Value> output_tensors;
    try {
        output_tensors = session_.Run(
            Ort::RunOptions{nullptr},
            input_names_.data(),
            &input_tensor,
            1,
            output_names_.data(),
            output_names_.size());
    } catch (...) {
        // fallback output name
        const char* alt_output_names[] = {"save_infer_model/scale_0.tmp_1"};
        output_tensors = session_.Run(
            Ort::RunOptions{nullptr},
            input_names_.data(),
            &input_tensor,
            1,
            alt_output_names,
            1);
    }

    if (output_tensors.empty() || !output_tensors[0].IsTensor()) return "";

    auto info = output_tensors[0].GetTensorTypeAndShapeInfo();
    auto shape = info.GetShape();
    const float* out = output_tensors[0].GetTensorData<float>();

    if (shape.size() == 3) {
        return decode_ctc(out, (int)shape[0], (int)shape[1], (int)shape[2]);
    } else if (shape.size() == 2) {
        return decode_ctc(out, 1, (int)shape[0], (int)shape[1]);
    }

    return "";

}
