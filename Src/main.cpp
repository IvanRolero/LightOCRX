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
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <filesystem>
#include <algorithm>
#include <fstream>

#include "text_detect.h"
#include "text_recon.h"
#include "pdf_export.h"


#include "stb_image.h"

namespace fs = std::filesystem;

static void print_help() {
    std::cout <<
R"(Usage: fullocr <detection_model.onnx> <recognition_model.onnx> <images...> [options]

Arguments:
  detection_model.onnx    Path to detection model
  recognition_model.onnx  Path to recognition model
  images...               One or more PNG/JPG image files or a directory

Options:
  -boxt             Box threshold (default: 0.6)
  -bint             Binarization threshold (default: 0.3)
  -unclip           Unclip ratio (default: 2.0)
  -targetsize <int> Target size for detection (default: 640)
  -savecrops <0/1>  Save cropped text regions (default: 0)
  -debugboxes <0/1> Draw detected boxes on an image (default: 0)
  -debugout <path>  Output path for debug image (default: detected_boxes.png)
  -dict <path>      Path to dictionary file (default: dict.txt)
  -targetw <int>    Target width for recognition (0 for dynamic, -1 to use crop aspect ratio)
  -targeth <int>    Target height for recognition (default: 48)
  -pdf <path>       Export as searchable PDF (optional, creates multi-page PDF for multiple images)
  -minconf <float>  Minimum confidence for PDF export (default: 0.3)
  -pdftext          Export PDF as text-only (white background, black visible text, no images)
  -text <path>      Export recognized text to a text file (plain text format)
  -textcls <0/1>    Enable/disable text orientation classification and 180 degrees rotation correction (default: 0)
  -clsmodel <path>  Path to rotation classification model (required if -textcls 1)
  -h, --help        Show this help message

Examples:
  lightocrx det.onnx rec.onnx image1.jpg image2.jpg -boxt 0.2 -dict mydict.txt
  lightocrx det.onnx rec.onnx *.jpg -savecrops 1 -debugboxes 1
  lightocrx det.onnx rec.onnx images/ -pdf output.pdf  # Process all images in directory
  lightocrx det.onnx rec.onnx image1.jpg image2.jpg -pdf output.pdf -pdftext  # Text-only PDF
  lightocrx det.onnx rec.onnx image1.jpg image2.jpg -text output.txt  # Export plain text
  lightocrx det.onnx rec.onnx image.jpg -textcls 1 -clsmodel cls.onnx  # Enable rotation correction
)";
}

static bool is_flag(const std::string& s) {
    return !s.empty() && s[0] == '-';
}

static std::vector<std::string> expand_directory(const std::string& path) {
    std::vector<std::string> image_files;
    
    try {
        if (fs::exists(path) && fs::is_directory(path)) {
            for (const auto& entry : fs::directory_iterator(path)) {
                if (fs::is_regular_file(entry.path())) {
                    std::string ext = entry.path().extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    
                    if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || 
                        ext == ".jfif" || ext == ".bmp" || ext == ".tiff" || ext == ".tif") {
                        image_files.push_back(entry.path().string());
                    }
                }
            }
            
            std::sort(image_files.begin(), image_files.end());
        } else if (fs::exists(path) && fs::is_regular_file(path)) {
            image_files.push_back(path);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error accessing path '" << path << "': " << e.what() << "\n";
    }
    
    return image_files;
}

static std::vector<std::string> collect_image_files(int argc, char** argv, int start_index) {
    std::vector<std::string> all_images;
    
    for (int i = start_index; i < argc; i++) {
        std::string arg = argv[i];
        
        if (is_flag(arg)) {
            break;
        }
        
        std::vector<std::string> files = expand_directory(arg);
        all_images.insert(all_images.end(), files.begin(), files.end());
    }
    
    std::sort(all_images.begin(), all_images.end());
    all_images.erase(std::unique(all_images.begin(), all_images.end()), all_images.end());
    
    return all_images;
}

// Function to export text to a plain text file
static bool export_text_file(
    const std::string& output_path,
    const std::vector<std::string>& image_paths,
    const std::vector<std::vector<OCRTextBox>>& all_pages_boxes,
    float min_confidence = 0.3f) {
    
    try {
        std::ofstream out_file(output_path);
        if (!out_file.is_open()) {
            std::cerr << "Error: Could not open text file for writing: " << output_path << "\n";
            return false;
        }
        
        int total_text_blocks = 0;
        
        for (size_t page_idx = 0; page_idx < image_paths.size(); page_idx++) {
            const std::string& image_path = image_paths[page_idx];
            const std::vector<OCRTextBox>& boxes = all_pages_boxes[page_idx];
            
            out_file << "--- Page " << (page_idx + 1) << ": " 
                     << fs::path(image_path).filename().string() << " ---\n";
            
            bool page_has_text = false;
            for (const auto& box : boxes) {
                if (!box.text.empty() && box.text_confidence >= min_confidence) {
                    out_file << box.text << "\n";
                    total_text_blocks++;
                    page_has_text = true;
                }
            }
            
            out_file << "\n";
            
            if (!page_has_text) {
                out_file << "[No text detected or all text below confidence threshold]\n\n";
            }
        }
        
        out_file.close();
        
        std::cout << "\n Saved text file: " << output_path << std::endl;
        std::cout << "  Total pages: " << image_paths.size() << std::endl;
        std::cout << "  Total text lines exported: " << total_text_blocks << std::endl;
        std::cout << "  Min confidence used: " << min_confidence << std::endl;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error exporting text file: " << e.what() << "\n";
        return false;
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        print_help();
        return 0;
    }

    std::string det_model = argv[1];
    std::string rec_model = argv[2];
    
    std::vector<std::string> image_paths = collect_image_files(argc, argv, 3);
    
    if (image_paths.empty()) {
        std::cerr << "Error: No valid image files found.\n";
        print_help();
        return 1;
    }

    DetectOptions dopt;
    ReconOptions ropt;
    std::string pdf_output;
    std::string text_output;
    float min_confidence = 0.3f;
    bool pdf_text_only = false;
    bool use_text_cls = false;        
    std::string cls_model_path;       
    
    int option_start = 3 + (int)image_paths.size();
    for (int i = option_start; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_help();
            return 0;
        } else if (arg == "-boxt" && i + 1 < argc) {
            dopt.box_thresh = std::stof(argv[++i]);
        } else if (arg == "-bint" && i + 1 < argc) {
            dopt.bin_thresh = std::stof(argv[++i]);
        } else if (arg == "-unclip" && i + 1 < argc) {
            dopt.unclip_ratio = std::stof(argv[++i]);
        } else if (arg == "-targetsize" && i + 1 < argc) {
            dopt.target_size = std::stoi(argv[++i]);
        } else if (arg == "-savecrops" && i + 1 < argc) {
            dopt.save_crops = (std::stoi(argv[++i]) != 0);
        } else if (arg == "-debugboxes" && i + 1 < argc) {
            dopt.debug_boxes = (std::stoi(argv[++i]) != 0);
        } else if (arg == "-debugout" && i + 1 < argc) {
            dopt.debug_output = argv[++i];
        } else if (arg == "-dict" && i + 1 < argc) {
            ropt.dict_path = argv[++i];
        } else if (arg == "-targetw" && i + 1 < argc) {
            ropt.target_w = std::stoi(argv[++i]);
        } else if (arg == "-targeth" && i + 1 < argc) {
            ropt.target_h = std::stoi(argv[++i]);
        } else if (arg == "-pdf" && i + 1 < argc) {
            pdf_output = argv[++i];
        } else if (arg == "-minconf" && i + 1 < argc) {
            min_confidence = std::stof(argv[++i]);
        } else if (arg == "-pdftext") {
            pdf_text_only = true;
        } else if (arg == "-text" && i + 1 < argc) {
            text_output = argv[++i];
        } else if (arg == "-textcls" && i + 1 < argc) {  
            use_text_cls = (std::stoi(argv[++i]) != 0);
        } else if (arg == "-clsmodel" && i + 1 < argc) { 
            cls_model_path = argv[++i];
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_help();
            return 1;
        }
    }

    // New: Validate that -clsmodel is provided when -textcls 1
    if (use_text_cls && cls_model_path.empty()) {
        std::cerr << "Error: -clsmodel <path> is required when -textcls 1 is enabled.\n";
        print_help();
        return 1;
    }

    // Set rotation classification options
    dopt.use_rotation_cls = use_text_cls;
    dopt.cls_model_path = cls_model_path;

    try {
        std::cout << "=======================================\n";
        std::cout << "Text Detection & Recognition\n";
        std::cout << "=======================================\n";
        std::cout << "Number of images to process: " << image_paths.size() << "\n";
        std::cout << "Detection model: " << det_model << "\n";
        std::cout << "Recognition model: " << rec_model << "\n";
        std::cout << "Detection target size: " << dopt.target_size << "\n";
        std::cout << "Rotation correction: " << (use_text_cls ? "Enabled" : "Disabled") << "\n";
        if (use_text_cls) {
            std::cout << "Rotation classification model: " << cls_model_path << "\n";
        }
        
        if (!pdf_output.empty()) {
            std::cout << "PDF output: " << pdf_output << " (multi-page)\n";
            std::cout << "Min confidence: " << min_confidence << "\n";
            if (pdf_text_only) {
                std::cout << "PDF mode: Text-only (white background, black visible text)\n";
            } else {
                std::cout << "PDF mode: Searchable overlay (image background, invisible text)\n";
            }
        }
        
        if (!text_output.empty()) {
            std::cout << "Text output: " << text_output << "\n";
            std::cout << "Min confidence: " << min_confidence << "\n";
        }
        
        std::cout << "---------------------------------------\n";
        
        TextDetector detector(det_model);
        TextRecognizer recognizer(rec_model, ropt);
        
        std::vector<std::string> pdf_image_paths;
        std::vector<int> pdf_image_widths;
        std::vector<int> pdf_image_heights;
        std::vector<std::vector<OCRTextBox>> all_pages_boxes;
        
        int total_text_regions = 0;
        int total_recognized = 0;
        
        for (size_t img_idx = 0; img_idx < image_paths.size(); img_idx++) {
            const std::string& image_path = image_paths[img_idx];
            
            std::cout << "\nProcessing image " << (img_idx + 1) << "/" << image_paths.size() 
                      << ": " << fs::path(image_path).filename().string() << "\n";
            
            int img_w = 0, img_h = 0, img_channels = 0;
            unsigned char* img_data = stbi_load(image_path.c_str(), &img_w, &img_h, &img_channels, 0);
            if (!img_data) {
                std::cerr << "Error: Could not load image: " << image_path << "\n";
                continue;
            }
            stbi_image_free(img_data);
            
            std::cout << "Image dimensions: " << img_w << "x" << img_h << "\n";
            std::cout << "Running text detection...\n";
            
            auto crops = detector.detect_and_crop(image_path, dopt);
            
            if (crops.empty()) {
                std::cout << "No text detected in this image.\n";
                
                if (!pdf_output.empty() || !text_output.empty()) {
                    pdf_image_paths.push_back(image_path);
                    pdf_image_widths.push_back(img_w);
                    pdf_image_heights.push_back(img_h);
                    all_pages_boxes.push_back(std::vector<OCRTextBox>());
                }
                continue;
            }
            
            std::cout << "Detected " << crops.size() << " text regions.\n";
            std::cout << "Running text recognition...\n";
            
            std::vector<OCRTextBox> ocr_boxes;
            ocr_boxes.reserve(crops.size());
            
            for (size_t i = 0; i < crops.size(); i++) {
                std::string text = recognizer.recognize(
                    crops[i].data.data(),
                    crops[i].w,
                    crops[i].h,
                    crops[i].c
                );

                size_t start = text.find_first_not_of(" \t\n\r");
                size_t end = text.find_last_not_of(" \t\n\r");
                if (start != std::string::npos && end != std::string::npos) {
                    text = text.substr(start, end - start + 1);
                }

                std::cout << "  [" << (i + 1) << "/" << crops.size() << "] ";
                if (!text.empty()) {
                    std::cout << "Text: \"" << text << "\"";
                    if (crops[i].box.score >= 0) {
                        std::cout << " (Confidence: " << static_cast<int>(crops[i].box.score * 100) << "%)";
                    }
                    std::cout << "\n";
                } else {
                    std::cout << "No text recognized\n";
                }
                
                OCRTextBox ocr_box;
                
                for (int j = 0; j < 8; j++) {
                    ocr_box.pts[j] = crops[i].box.pts[j];
                }
                ocr_box.score = crops[i].box.score;
                ocr_box.text = text;
                ocr_box.text_confidence = crops[i].box.score;
                
                ocr_boxes.push_back(ocr_box);
            }
            
            total_text_regions += (int)crops.size();
            for (const auto& box : ocr_boxes) {
                if (!box.text.empty()) total_recognized++;
            }
            
            if (!pdf_output.empty() || !text_output.empty()) {
                pdf_image_paths.push_back(image_path);
                pdf_image_widths.push_back(img_w);
                pdf_image_heights.push_back(img_h);
                all_pages_boxes.push_back(ocr_boxes);
            }
        }
        
        if (!text_output.empty() && !pdf_image_paths.empty()) {
            std::cout << "\n---------------------------------------\n";
            std::cout << "Exporting recognized text to file...\n";
            
            bool success = export_text_file(
                text_output,
                pdf_image_paths,
                all_pages_boxes,
                min_confidence);
            
            if (!success) {
                std::cerr << "Warning: Text export failed.\n";
            }
        }
        
        if (!pdf_output.empty() && !pdf_image_paths.empty()) {
            std::cout << "\n---------------------------------------\n";
            std::cout << "Exporting to multi-page searchable PDF...\n";
            
            PDFExporter pdf_exporter;
            bool success = pdf_exporter.createMultiPagePDF(
                pdf_output, 
                pdf_image_paths, 
                pdf_image_widths, 
                pdf_image_heights, 
                all_pages_boxes, 
                min_confidence,
                pdf_text_only);
            
            if (!success) {
                std::cerr << "Warning: PDF export failed.\n";
            }
        }
        
        std::cout << "\n=======================================\n";
        std::cout << "Processing Complete\n";
        std::cout << "=======================================\n";
        std::cout << "Total images processed: " << image_paths.size() << "\n";
        std::cout << "Total text regions detected: " << total_text_regions << "\n";
        std::cout << "Successfully recognized: " << total_recognized << "\n";
        
        if (!pdf_output.empty()) {
            std::cout << "Searchable PDF created: " << pdf_output << "\n";
            std::cout << "Total pages in PDF: " << pdf_image_paths.size() << "\n";
            if (pdf_text_only) {
                std::cout << "PDF contains black visible text on white background (no images).\n";
            } else {
                std::cout << "PDF contains invisible searchable text overlaid on original images.\n";
            }
        }
        
        if (!text_output.empty()) {
            std::cout << "Text file created: " << text_output << "\n";
            std::cout << "Contains recognized text from all images with page separation.\n";
        }
        
        std::cout << "=======================================\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n=======================================\n";
        std::cerr << "Error: " << e.what() << "\n";
        std::cerr << "=======================================\n";
        return 1;
    }

}
