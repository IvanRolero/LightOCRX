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
#include <vector>
#include <string>
#include <cstdint>
#include <memory>
#include "include/onnxruntime_cxx_api.h"

struct TextBox {
    float pts[8];  // x1,y1,x2,y2,x3,y3,x4,y4
    float score;
};

struct CropResult {
    TextBox box;
    int w = 0;
    int h = 0;
    int c = 0;
    std::vector<uint8_t> data; // cropped image pixels (RGB or RGBA)
};

struct DetectOptions {
    float bin_thresh = 0.3f;
    float box_thresh = 0.6f;
    float unclip_ratio = 2.0f;
    int min_component_area = 50;
    bool save_crops = false;
    bool debug_boxes = false;  // Debug flag for box visualization
    std::string debug_output = "detected_boxes.png";  // Output file for debug
    int target_size = 640;
    bool use_rotation_cls = false; 
    std::string cls_model_path = ""; 
};

// Forward declaration so TextDetector can hold a cached rotation classifier
// without exposing ONNX Runtime details in this header.
class RotationCLSModel;

class TextDetector {
public:
    explicit TextDetector(const std::string& det_model_path);
    ~TextDetector();

    // Runs detection on input image path and returns crops + box info.
    // If out_img_w / out_img_h are non-null, they receive the loaded image's
    // dimensions so callers don't need to load the image a second time just
    // to learn its size.
    std::vector<CropResult> detect_and_crop(const std::string& image_path,
                                           const DetectOptions& opt,
                                           int* out_img_w = nullptr,
                                           int* out_img_h = nullptr);

private:
    std::string model_path_;
    
    // ONNX Runtime objects
    Ort::Env env_{nullptr};
    Ort::Session session_{nullptr};
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<std::string> input_names_str_;
    std::vector<std::string> output_names_str_;

    // Cached rotation classifier so we don't reload the .onnx from disk on
    // every image. Loaded lazily and reused while cls_model_path is unchanged.
    std::unique_ptr<RotationCLSModel> cls_model_;
    std::string cls_model_path_loaded_;

};
