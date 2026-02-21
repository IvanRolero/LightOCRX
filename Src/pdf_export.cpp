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
#include "pdf_export.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cfloat>
#include <filesystem>

// Math constants
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace fs = std::filesystem;

void PDFExporter::error_handler(HPDF_STATUS error_no, HPDF_STATUS detail_no, void* user_data) {
    if (error_no >= 4000) { 
        std::cerr << "PDF Error: " << error_no << " Detail: " << detail_no << std::endl;
    }
}

PDFExporter::PDFExporter() {}
PDFExporter::~PDFExporter() {}

// Single page version (backward compatible)
bool PDFExporter::createSearchablePDF(
    const std::string& pdf_path,
    const std::string& image_path,
    int img_w, int img_h,
    const std::vector<OCRTextBox>& boxes,
    float min_confidence,
    bool text_only) {
    
    return createMultiPagePDF(pdf_path, 
                             std::vector<std::string>{image_path},
                             std::vector<int>{img_w},
                             std::vector<int>{img_h},
                             std::vector<std::vector<OCRTextBox>>{boxes},
                             min_confidence,
                             text_only);
}

// Multi-page PDF creation
bool PDFExporter::createMultiPagePDF(
    const std::string& pdf_path,
    const std::vector<std::string>& image_paths,
    const std::vector<int>& img_widths,
    const std::vector<int>& img_heights,
    const std::vector<std::vector<OCRTextBox>>& pages_boxes,
    float min_confidence,
    bool text_only) {
    
    // Validate input
    if (image_paths.empty() || 
        image_paths.size() != img_widths.size() || 
        image_paths.size() != img_heights.size() ||
        image_paths.size() != pages_boxes.size()) {
        std::cerr << "Error: Invalid input sizes for multi-page PDF\n";
        return false;
    }
    
    std::cout << "\nCreating multi-page searchable PDF (" << image_paths.size() << " pages)..." << std::endl;
    if (text_only) {
        std::cout << "  (Exporting as TEXT-ONLY PDF with visible black text)\n";
    }
    
    HPDF_Doc pdf = HPDF_New(error_handler, nullptr);
    if (!pdf) {
        std::cerr << "Failed to create PDF object\n";
        return false;
    }

    // Enable UTF-8 encoding support
    HPDF_UseUTFEncodings(pdf);
    HPDF_SetCurrentEncoder(pdf, "UTF-8");
    
    // Set PDF properties
    HPDF_SetInfoAttr(pdf, HPDF_INFO_CREATOR, "LightOCRX");
    HPDF_SetInfoAttr(pdf, HPDF_INFO_TITLE, "OCR Results");
    HPDF_SetInfoAttr(pdf, HPDF_INFO_AUTHOR, "LightOCRX");
    
    // Load Unicode font (NotoSans) with proper name detection
    HPDF_Font font = nullptr;
    const char* font_path = "NotoSans-Regular.ttf";
    
    // Check if font file exists before attempting to load
    if (std::filesystem::exists(font_path)) {
        // Load the font and get the internal font name
        const char* loaded_font_name = HPDF_LoadTTFontFromFile(pdf, font_path, HPDF_TRUE);
        
        if (loaded_font_name) {
            std::cout << "  Font file loaded, registered as: '" << loaded_font_name << "'" << std::endl;
            
            // Try to get the font with its registered name
            font = HPDF_GetFont(pdf, loaded_font_name, "UTF-8");
            
            if (font) {
                std::cout << "  Successfully loaded Unicode font: " << loaded_font_name << std::endl;
            } else {
                std::cerr << "  Warning: Could not get font with name '" << loaded_font_name << "'" << std::endl;
            }
        } else {
            std::cerr << "  Warning: Failed to load font from file: " << font_path << std::endl;
        }
    } else {
        std::cout << "  Note: Font file not found at: " << font_path << std::endl;
        std::cout << "  Will use built-in Helvetica font (ASCII only)" << std::endl;
    }
    
    // Fallback to Helvetica if Unicode font loading failed
    if (!font) {
        font = HPDF_GetFont(pdf, "Helvetica", nullptr);
        if (font) {
            std::cout << "  Using Helvetica fallback font (ASCII text only)" << std::endl;
            std::cout << "  Note: Non-ASCII characters may not display correctly" << std::endl;
        } else {
            std::cerr << "  ERROR: Could not load any font!" << std::endl;
            HPDF_Free(pdf);
            return false;
        }
    }
    
    int total_text_blocks = 0;
    int total_skipped_boxes = 0;
    
    // Create pages
    for (size_t page_idx = 0; page_idx < image_paths.size(); page_idx++) {
        const std::string& image_path = image_paths[page_idx];
        int img_w = img_widths[page_idx];
        int img_h = img_heights[page_idx];
        const std::vector<OCRTextBox>& boxes = pages_boxes[page_idx];
        
        std::cout << "  Adding page " << (page_idx + 1) << "/" << image_paths.size() 
                  << ": " << fs::path(image_path).filename().string() 
                  << " (" << img_w << "x" << img_h << ")" << std::endl;
        
        HPDF_Page page = HPDF_AddPage(pdf);
        if (!page) {
            std::cerr << "  Error: Failed to add page " << (page_idx + 1) << std::endl;
            continue;
        }
        
        HPDF_Page_SetWidth(page, (float)img_w);
        HPDF_Page_SetHeight(page, (float)img_h);
        
        // Add background image only if NOT text-only mode
        if (!text_only) {
            HPDF_Image img = nullptr;
            
            // Get file extension to determine image type
            fs::path image_fs_path(image_path);
            std::string extension = image_fs_path.extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
            
            // Try to load image based on extension
            if (extension == ".png") {
                img = HPDF_LoadPngImageFromFile(pdf, image_path.c_str());
            } else if (extension == ".jpg" || extension == ".jpeg" || 
                       extension == ".jfif" || extension == ".jpe") {
                img = HPDF_LoadJpegImageFromFile(pdf, image_path.c_str());
            } else {
                // Try both PNG and JPEG for unknown extensions
                img = HPDF_LoadPngImageFromFile(pdf, image_path.c_str());
                if (!img) {
                    img = HPDF_LoadJpegImageFromFile(pdf, image_path.c_str());
                }
            }
            
            if (img) {
                try {
                    if (HPDF_Page_DrawImage(page, img, 0, 0, (float)img_w, (float)img_h) != HPDF_OK) {
                        std::cout << "    Warning: Could not draw image, using white background" << std::endl;
                        HPDF_Page_SetRGBFill(page, 1.0, 1.0, 1.0);
                        HPDF_Page_Rectangle(page, 0, 0, (float)img_w, (float)img_h);
                        HPDF_Page_Fill(page);
                    } else {
                        std::cout << "    Background image loaded successfully" << std::endl;
                    }
                } catch (...) {
                    std::cout << "    Warning: Exception drawing image, using white background" << std::endl;
                    HPDF_Page_SetRGBFill(page, 1.0, 1.0, 1.0);
                    HPDF_Page_Rectangle(page, 0, 0, (float)img_w, (float)img_h);
                    HPDF_Page_Fill(page);
                }
            } else {
                std::cout << "    Warning: Could not load background image, using white background" << std::endl;
                HPDF_Page_SetRGBFill(page, 1.0, 1.0, 1.0);
                HPDF_Page_Rectangle(page, 0, 0, (float)img_w, (float)img_h);
                HPDF_Page_Fill(page);
            }
        } else {
            // In text-only mode, use white background
            HPDF_Page_SetRGBFill(page, 1.0, 1.0, 1.0);
            HPDF_Page_Rectangle(page, 0, 0, (float)img_w, (float)img_h);
            HPDF_Page_Fill(page);
        }
        
        int text_blocks_added = 0;
        int page_skipped_boxes = 0;
        
        // Process boxes for this page
        for (const auto& b : boxes) {
            if (b.text.empty() || b.text_confidence < min_confidence) continue;
            
            // Wrap each box in a try-catch to prevent corruption
            try {
                // --- GEOMETRY CALCULATION ---
                
                // PaddleOCR Points: 0:TL, 1:TR, 2:BR, 3:BL
                float bl_x = b.pts[6];
                float bl_y = b.pts[7];
                float br_x = b.pts[4];
                float br_y = b.pts[5];
                float tr_x = b.pts[2];
                float tr_y = b.pts[3];
                float tl_x = b.pts[0];
                float tl_y = b.pts[1];
                
                // Validate points are finite and within reasonable range
                bool valid_points = true;
                float pts_array[8] = {tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y};
                for (int i = 0; i < 8; i++) {
                    if (std::isnan(pts_array[i]) || std::isinf(pts_array[i]) || 
                        pts_array[i] < -100000 || pts_array[i] > 100000) {
                        valid_points = false;
                        break;
                    }
                }
                
                if (!valid_points) {
                    page_skipped_boxes++;
                    continue;
                }
                
                // Convert Y to PDF coordinates (Flip Y)
                float pdf_bl_y = (float)img_h - bl_y;
                float pdf_br_y = (float)img_h - br_y;
                float pdf_tl_y = (float)img_h - tl_y;
                
                // Calculate Box Dimensions
                float dx = br_x - bl_x;
                float dy = pdf_br_y - pdf_bl_y;
                float box_width = std::sqrt(dx*dx + dy*dy);
                
                // Height = Distance between Top-Left and Bottom-Left
                float h_dx = tl_x - bl_x;
                float h_dy = pdf_tl_y - pdf_bl_y;
                float box_height = std::sqrt(h_dx*h_dx + h_dy*h_dy);
                
                // --- IMPROVED VALIDATION ---
                // Check for NaN or Inf values
                if (std::isnan(box_width) || std::isnan(box_height) ||
                    std::isinf(box_width) || std::isinf(box_height)) {
                    page_skipped_boxes++;
                    continue;
                }
                
                // Check for degenerate boxes (too small)
                const float MIN_BOX_SIZE = 2.0f;  // Minimum 2 pixels for safety
                const float MAX_BOX_SIZE = 5000.0f;  // Sanity cap
                
                if (box_width < MIN_BOX_SIZE || box_height < MIN_BOX_SIZE) {
                    page_skipped_boxes++;
                    continue;
                }
                
                // Cap extremely large boxes
                if (box_width > MAX_BOX_SIZE) box_width = MAX_BOX_SIZE;
                if (box_height > MAX_BOX_SIZE) box_height = MAX_BOX_SIZE;
                
                // Ensure minimum font size for libharu
                float min_font_size = 2.0f;
                float max_font_size = 500.0f;
                
                if (box_height < min_font_size) {
                    box_height = min_font_size;
                }
                if (box_height > max_font_size) {
                    box_height = max_font_size;
                }
                
                // Round font size to nearest integer (libharu prefers this)
                box_height = std::round(box_height);
                
                // Calculate Rotation Angle
                float angle = std::atan2(dy, dx);
                
                // --- TEXT FITTING ---
                
                // Set Font Size to Box Height - check return value
                if (HPDF_Page_SetFontAndSize(page, font, box_height) != HPDF_OK) {
                    page_skipped_boxes++;
                    continue;
                }
                
                // Calculate Horizontal Scaling
                float text_real_width = HPDF_Page_TextWidth(page, b.text.c_str());
                
                if (text_real_width < 0.1f) {
                    page_skipped_boxes++;
                    continue;
                }
                
                float h_scale = (box_width / text_real_width) * 100.0f;
                
                // Clamp scale to reasonable values
                const float MIN_SCALE = 10.0f;
                const float MAX_SCALE = 300.0f;
                
                if (h_scale < MIN_SCALE) h_scale = MIN_SCALE;
                if (h_scale > MAX_SCALE) h_scale = MAX_SCALE;
                if (std::isnan(h_scale) || std::isinf(h_scale)) {
                    h_scale = 100.0f;
                }
                
                // Start text block
                if (HPDF_Page_BeginText(page) != HPDF_OK) {
                    page_skipped_boxes++;
                    continue;
                }
                
                // Set text rendering mode based on text_only flag
                if (text_only) {
                    // In text-only mode: visible black text
                    HPDF_Page_SetTextRenderingMode(page, HPDF_FILL);
                    HPDF_Page_SetRGBFill(page, 0.0, 0.0, 0.0); // Black text
                } else {
                    // Normal mode: invisible text for overlay
                    HPDF_Page_SetTextRenderingMode(page, HPDF_INVISIBLE);
                }
                
                HPDF_Page_SetHorizontalScalling(page, h_scale);
                
                // Create Transformation Matrix
                float cos_a = std::cos(angle);
                float sin_a = std::sin(angle);
                
                // Validate matrix values
                if (std::isnan(cos_a) || std::isnan(sin_a) ||
                    std::isinf(cos_a) || std::isinf(sin_a)) {
                    HPDF_Page_EndText(page);
                    page_skipped_boxes++;
                    continue;
                }
                
                if (HPDF_Page_SetTextMatrix(page,
                    cos_a, sin_a, -sin_a, cos_a,
                    bl_x, pdf_bl_y) != HPDF_OK) {
                    HPDF_Page_EndText(page);
                    page_skipped_boxes++;
                    continue;
                }
                
                // Show text directly (UTF-8)
                if (HPDF_Page_ShowText(page, b.text.c_str()) != HPDF_OK) {
                    HPDF_Page_EndText(page);
                    page_skipped_boxes++;
                    continue;
                }
                
                if (HPDF_Page_EndText(page) != HPDF_OK) {
                    page_skipped_boxes++;
                    continue;
                }
                
                text_blocks_added++;
                total_text_blocks++;
                
            } catch (const std::exception& e) {
                // Make sure we end text if we started it
                try {
                    HPDF_Page_EndText(page);
                } catch (...) {}
                page_skipped_boxes++;
                continue;
            } catch (...) {
                // Catch any other exceptions
                try {
                    HPDF_Page_EndText(page);
                } catch (...) {}
                page_skipped_boxes++;
                continue;
            }
        }
        
        std::cout << "Added " << text_blocks_added << "text blocks to this page.";
        if (page_skipped_boxes > 0) {
            std::cout << "(Skipped " << page_skipped_boxes << " invalid/malformed boxes)";
            total_skipped_boxes += page_skipped_boxes;
        }
        std::cout << std::endl;
    }
    
    // Save the PDF only if we have content
    if (total_text_blocks > 0 || !text_only) {
        HPDF_STATUS save_status = HPDF_SaveToFile(pdf, pdf_path.c_str());
        if (save_status != HPDF_OK) {
            std::cerr << "Error: Failed to save PDF file (status: " << save_status << ")\n";
            HPDF_Free(pdf);
            return false;
        }
    } else {
        std::cerr << "Error: No content to save in PDF\n";
        HPDF_Free(pdf);
        return false;
    }
    
    HPDF_Free(pdf);
    
    std::cout << "\n Saved multi-page searchable OCR PDF: " << pdf_path << std::endl;
    std::cout << "  Total pages: " << image_paths.size() << std::endl;
    std::cout << "  Total text blocks: " << total_text_blocks << std::endl;
    if (total_skipped_boxes > 0) {
        std::cout << "  Skipped invalid boxes: " << total_skipped_boxes << std::endl;
    }
    if (text_only) {
        std::cout << "  Mode: Text-only (white background, black visible text)\n";
    } else {
        std::cout << "  Mode: Searchable overlay (image background, invisible text)\n";
    }
    std::cout << "========================================\n" << std::endl;
    
    return true;
}
