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

#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include "include/onnxruntime_cxx_api.h"

struct ReconOptions {
    std::string dict_path = "dict.txt";
    int target_h = 48;
    int target_w = 0;  
    int max_dict_size = 30000;
    bool debug_boxes = false; 
};

class TextRecognizer {
public:
    TextRecognizer(const std::string& rec_model_path, const ReconOptions& opt);

    // Recognize from an in-memory crop image (RGB/RGBA)
    std::string recognize(const uint8_t* data, int w, int h, int c);
    
    // Helper to compute dynamic width based on aspect ratio
    static int compute_dynamic_width(int h, int w, int base_h = 48, int min_w = 256, int max_w = 4096);

private:
    std::string decode_ctc(const float* output, int batch, int time_step, int class_num);
    void load_dict_or_default(const std::string& dict_path);

private:
    ReconOptions opt_;
    std::vector<std::string> char_dict_;

    Ort::Env env_;
    Ort::Session session_{nullptr};

    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<std::string> input_names_str_;
    std::vector<std::string> output_names_str_;

};

