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
#include <hpdf.h>

#ifdef _WIN32
#include <windows.h>
#include <stringapiset.h>
#endif

// Enhanced TextBox structure with OCR results
struct OCRTextBox {
    float pts[8];  // x1,y1,x2,y2,x3,y3,x4,y4 - EXACT SAME FORMAT as TextBox
    float score;   // Detection confidence
    std::string text;      // Recognized text (UTF-8)
    float text_confidence; // Recognition confidence
};

class PDFExporter {
public:
    PDFExporter();
    ~PDFExporter();
    
    // Single page version (backward compatible)
    bool createSearchablePDF(
        const std::string& pdf_path,
        const std::string& image_path,
        int img_w, int img_h,
        const std::vector<OCRTextBox>& boxes,
        float min_confidence = 0.3f,
        bool text_only = false);
    
    // Multi-page version
    bool createMultiPagePDF(
        const std::string& pdf_path,
        const std::vector<std::string>& image_paths,
        const std::vector<int>& img_widths,
        const std::vector<int>& img_heights,
        const std::vector<std::vector<OCRTextBox>>& pages_boxes,
        float min_confidence = 0.3f,
        bool text_only = false);
    
private:
    // UTF-8 to WinAnsi conversion for Windows
    std::string utf8_to_winansi(const std::string& utf8_str);
    
    // Error handler for libharu
    static void error_handler(HPDF_STATUS error_no, HPDF_STATUS detail_no, void* user_data);

};
