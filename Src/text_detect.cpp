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

#include "text_detect.h"

#include "include/onnxruntime_cxx_api.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cfloat>
#include <queue>
#include <fstream>
#include <memory>
#include <codecvt>
#include <locale>
#include <filesystem>  // C++17 filesystem

#include "stb_image.h"
#include "stb_image_write.h"
#include "stb_image_resize2.h"
#include "clipper.hpp"

#ifdef _WIN32
#include <windows.h>
#include <stringapiset.h>
static std::wstring to_wstring(const std::string& s) {
    int len = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    std::wstring ws(len, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &ws[0], len);
    return ws;
}
#endif

using namespace ClipperLib;

struct Point {
    int x, y;
    Point(int x = 0, int y = 0) : x(x), y(y) {}
};

struct PointF {
    float x, y;
    PointF(float x = 0, float y = 0) : x(x), y(y) {}
};


// Rotation Classification Model Class
class RotationCLSModel {
private:
    const int TARGET_H = 80;  // Typical PaddleCLS input size
    const int TARGET_W = 160;
    
    std::vector<std::string> labels_ = {"normal", "rotated_180"};
    Ort::Env env_;
    Ort::Session session_{nullptr};
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<std::string> input_names_str_;
    std::vector<std::string> output_names_str_;
    
    // Normalization parameters (ImageNet stats)
    const float mean_[3] = {0.485f, 0.456f, 0.406f};
    const float std_[3] = {0.229f, 0.224f, 0.225f};

public:
    RotationCLSModel(const std::string& model_path)
        : env_(ORT_LOGGING_LEVEL_WARNING, "ppcls_rotation") {

        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef _WIN32
        std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
        std::wstring wpath = converter.from_bytes(model_path);
        session_ = Ort::Session(env_, wpath.c_str(), opts);
#else
        session_ = Ort::Session(env_, model_path.c_str(), opts);
#endif

        Ort::AllocatorWithDefaultOptions allocator;

        for (size_t i = 0; i < session_.GetInputCount(); i++) {
            auto name = session_.GetInputNameAllocated(i, allocator);
            input_names_str_.push_back(name.get());
            input_names_.push_back(input_names_str_.back().c_str());
        }

        for (size_t i = 0; i < session_.GetOutputCount(); i++) {
            auto name = session_.GetOutputNameAllocated(i, allocator);
            output_names_str_.push_back(name.get());
            output_names_.push_back(output_names_str_.back().c_str());
        }
    }

    std::vector<float> preprocess(const std::vector<uint8_t>& img_data, int w, int h, int c,
                                  std::vector<int64_t>& input_shape) {
        // Convert to RGB if needed
        std::vector<uint8_t> rgb_data;
        if (c >= 3) {
            rgb_data.resize(w * h * 3);
            for (int i = 0; i < w * h; i++) {
                rgb_data[i * 3] = img_data[i * c];
                rgb_data[i * 3 + 1] = img_data[i * c + 1];
                rgb_data[i * 3 + 2] = img_data[i * c + 2];
            }
        } else {
            rgb_data.resize(w * h * 3);
            for (int i = 0; i < w * h; i++) {
                uint8_t val = img_data[i];
                rgb_data[i * 3] = val;
                rgb_data[i * 3 + 1] = val;
                rgb_data[i * 3 + 2] = val;
            }
        }

        // Resize to target dimensions (stretch to fit exactly)
        std::vector<float> input(3 * TARGET_H * TARGET_W);
        
        for (int y = 0; y < TARGET_H; y++) {
            int sy = y * h / TARGET_H;
            sy = std::min(sy, h - 1);
            
            for (int x = 0; x < TARGET_W; x++) {
                int sx = x * w / TARGET_W;
                sx = std::min(sx, w - 1);
                
                // Paddle models often expect BGR order (channel 2,1,0)
                for (int channel = 0; channel < 3; channel++) {
                    // Reverse channel order: RGB -> BGR
                    int src_channel = 2 - channel;
                    int src_idx = (sy * w + sx) * 3 + src_channel;
                    
                    // Normalize: (x/255 - mean) / std
                    float pixel = rgb_data[src_idx] / 255.0f;
                    float normalized = (pixel - mean_[channel]) / std_[channel];
                    
                    // CHW format
                    int dst_idx = channel * TARGET_H * TARGET_W + y * TARGET_W + x;
                    input[dst_idx] = normalized;
                }
            }
        }

        input_shape = {1, 3, TARGET_H, TARGET_W};
        return input;
    }

    int run(const std::vector<uint8_t>& img_data, int w, int h, int c) {
        std::vector<int64_t> input_shape;
        auto input_data = preprocess(img_data, w, h, c, input_shape);

        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem, input_data.data(), input_data.size(),
            input_shape.data(), input_shape.size());

        auto outputs = session_.Run(
            Ort::RunOptions{nullptr},
            input_names_.data(),
            &input_tensor,
            1,
            output_names_.data(),
            output_names_.size());

        if (outputs.empty()) {
            throw std::runtime_error("No outputs from rotation classification model");
        }

        const float* out_data = outputs[0].GetTensorData<float>();
        auto type_info = outputs[0].GetTensorTypeAndShapeInfo();
        size_t num_elements = type_info.GetElementCount();
        
        // Apply softmax for better probability interpretation
        std::vector<float> probs(num_elements);
        float max_val = *std::max_element(out_data, out_data + num_elements);
        float sum = 0.0f;
        
        for (size_t i = 0; i < num_elements; i++) {
            probs[i] = std::exp(out_data[i] - max_val);
            sum += probs[i];
        }
        
        for (size_t i = 0; i < num_elements; i++) {
            probs[i] /= sum;
        }
        
        // Find class with highest probability
        int max_idx = 0;
        float max_prob = probs[0];
        for (size_t i = 1; i < num_elements; i++) {
            if (probs[i] > max_prob) {
                max_prob = probs[i];
                max_idx = i;
            }
        }
        
        // Optional: Add confidence threshold
        const float CONFIDENCE_THRESHOLD = 0.6f;
        if (max_prob < CONFIDENCE_THRESHOLD) {
            // Low confidence, assume normal
            return 0;
        }

        // Return 0 for normal (upright), 180 for rotated
        return (max_idx == 0) ? 0 : 180;
    }
};


// Original DBNet detection functions

static float box_score_fast(const std::vector<float>& prob, int width, int height,
                            const std::vector<Point>& contour) {
    if (contour.size() < 3) return 0.0f;

    int min_x = width, max_x = 0, min_y = height, max_y = 0;
    for (const auto& p : contour) {
        min_x = std::min(min_x, p.x);
        max_x = std::max(max_x, p.x);
        min_y = std::min(min_y, p.y);
        max_y = std::max(max_y, p.y);
    }

    min_x = std::max(0, min_x);
    max_x = std::min(width - 1, max_x);
    min_y = std::max(0, min_y);
    max_y = std::min(height - 1, max_y);

    if (max_x <= min_x || max_y <= min_y) return 0.0f;

    auto point_in_polygon = [&](int x, int y) -> bool {
        bool inside = false;
        size_t n = contour.size();
        for (size_t i = 0, j = n - 1; i < n; j = i++) {
            if (((contour[i].y > y) != (contour[j].y > y)) &&
                (x < (contour[j].x - contour[i].x) * (y - contour[i].y) /
                             (float)(contour[j].y - contour[i].y) +
                         contour[i].x)) {
                inside = !inside;
            }
        }
        return inside;
    };

    float sum = 0.0f;
    int count = 0;

    for (int y = min_y; y <= max_y; y++) {
        for (int x = min_x; x <= max_x; x++) {
            if (point_in_polygon(x, y)) {
                sum += prob[y * width + x];
                count++;
            }
        }
    }

    return count > 0 ? sum / count : 0.0f;
}


static std::vector<std::vector<Point>> unclip_polygon(
    const std::vector<Point>& contour,
    float unclip_ratio) {
    if (contour.size() < 3) return {};

    double area = 0.0;
    int n = (int)contour.size();
    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        area += contour[i].x * contour[j].y - contour[j].x * contour[i].y;
    }
    area = std::abs(area) * 0.5;

    double perimeter = 0.0;
    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        double dx = contour[j].x - contour[i].x;
        double dy = contour[j].y - contour[i].y;
        perimeter += std::sqrt(dx * dx + dy * dy);
    }

    if (perimeter < 1e-6) return {};

    double distance = area * unclip_ratio / perimeter;

    Path src;
    for (const auto& p : contour) {
        src << IntPoint(p.x, p.y);
    }

    ClipperOffset offset;
    offset.AddPath(src, jtRound, etClosedPolygon);

    Paths dst;
    offset.Execute(dst, distance);

    std::vector<std::vector<Point>> results;
    for (const auto& path : dst) {
        if (path.size() < 3) continue;

        std::vector<Point> poly;
        for (const auto& p : path) {
            poly.emplace_back((int)p.X, (int)p.Y);
        }
        results.push_back(poly);
    }
    return results;
}


static std::vector<std::vector<Point>> find_connected_components(const std::vector<uint8_t>& bitmap,
                                                                 int width, int height,
                                                                 int min_area = 50) {
    std::vector<std::vector<Point>> components;
    std::vector<bool> visited(width * height, false);

    int dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    int dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            if (bitmap[idx] > 0 && !visited[idx]) {
                std::vector<Point> component;
                std::queue<Point> q;
                q.push(Point(x, y));
                visited[idx] = true;

                while (!q.empty()) {
                    Point p = q.front();
                    q.pop();
                    component.push_back(p);

                    for (int d = 0; d < 8; d++) {
                        int nx = p.x + dx[d];
                        int ny = p.y + dy[d];
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            int nidx = ny * width + nx;
                            if (bitmap[nidx] > 0 && !visited[nidx]) {
                                q.push(Point(nx, ny));
                                visited[nidx] = true;
                            }
                        }
                    }
                }

                if ((int)component.size() >= min_area) {
                    components.push_back(component);
                }
            }
        }
    }

    return components;
}


static std::vector<Point> convex_hull(std::vector<Point> points) {
    if (points.size() < 3) return points;

    int min_idx = 0;
    for (size_t i = 1; i < points.size(); i++) {
        if (points[i].y < points[min_idx].y ||
            (points[i].y == points[min_idx].y && points[i].x < points[min_idx].x)) {
            min_idx = (int)i;
        }
    }

    std::swap(points[0], points[min_idx]);
    Point p0 = points[0];

    std::sort(points.begin() + 1, points.end(),
              [p0](const Point& a, const Point& b) {
                  int cross = (a.x - p0.x) * (b.y - p0.y) - (a.y - p0.y) * (b.x - p0.x);
                  if (cross == 0) {
                      int da = (a.x - p0.x) * (a.x - p0.x) + (a.y - p0.y) * (a.y - p0.y);
                      int db = (b.x - p0.x) * (b.x - p0.x) + (b.y - p0.y) * (b.y - p0.y);
                      return da < db;
                  }
                  return cross > 0;
              });

    std::vector<Point> hull;
    hull.push_back(points[0]);

    for (size_t i = 1; i < points.size(); i++) {
        while (hull.size() >= 2) {
            Point& a = hull[hull.size() - 2];
            Point& b = hull.back();
            Point& c = points[i];

            int cross = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
            if (cross <= 0) hull.pop_back();
            else break;
        }
        hull.push_back(points[i]);
    }

    return hull;
}


static float polygon_area(const std::vector<Point>& poly) {
    if (poly.size() < 3) return 0.0f;

    double area = 0.0;
    int n = (int)poly.size();
    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        area += poly[i].x * poly[j].y - poly[j].x * poly[i].y;
    }
    return (float)(std::abs(area) * 0.5);
}


static std::vector<PointF> min_area_rect(const std::vector<Point>& poly) {
    if (poly.size() < 3) return {};

    std::vector<Point> hull = convex_hull(poly);
    if (hull.size() < 3) return {};

    int n = (int)hull.size();
    float best_area = FLT_MAX;
    std::vector<PointF> best_rect(4);

    std::vector<float> angles(n);
    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        float dx = (float)(hull[j].x - hull[i].x);
        float dy = (float)(hull[j].y - hull[i].y);
        angles[i] = atan2(dy, dx);
    }

    for (int i = 0; i < n; i++) {
        float angle = angles[i];
        float cos_a = cos(-angle);
        float sin_a = sin(-angle);

        float min_x = FLT_MAX, max_x = -FLT_MAX;
        float min_y = FLT_MAX, max_y = -FLT_MAX;

        for (const auto& p : hull) {
            float rx = p.x * cos_a - p.y * sin_a;
            float ry = p.x * sin_a + p.y * cos_a;
            min_x = std::min(min_x, rx);
            max_x = std::max(max_x, rx);
            min_y = std::min(min_y, ry);
            max_y = std::max(max_y, ry);
        }

        float area = (max_x - min_x) * (max_y - min_y);
        if (area < best_area) {
            best_area = area;

            std::vector<PointF> corners = {
                {min_x, min_y},
                {max_x, min_y},
                {max_x, max_y},
                {min_x, max_y},
            };

            cos_a = cos(angle);
            sin_a = sin(angle);
            for (int j = 0; j < 4; j++) {
                float x = corners[j].x * cos_a - corners[j].y * sin_a;
                float y = corners[j].x * sin_a + corners[j].y * cos_a;
                best_rect[j] = {x, y};
            }
        }
    }

    return best_rect;
}


// Helper function to calculate distance between two points
static float dist(const PointF& p1, const PointF& p2) {
    return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
}


// Function to extract and dewarp text regions using perspective transform
static void GetRotateCropImage(const uint8_t* src_data, int src_w, int src_h, int channels,
                               const std::vector<PointF>& box_points, 
                               std::vector<uint8_t>& dst_data, int& dst_w, int& dst_h) {
    
    // 1. Get the 4 points (Order should be: TL, TR, BR, BL)
    float x0 = box_points[0].x; float y0 = box_points[0].y;
    float x1 = box_points[1].x; float y1 = box_points[1].y;
    float x2 = box_points[2].x; float y2 = box_points[2].y;
    float x3 = box_points[3].x; float y3 = box_points[3].y;

    // 2. Calculate width and height of the dewarped box
    float w1 = dist(PointF(x0, y0), PointF(x1, y1));
    float w2 = dist(PointF(x2, y2), PointF(x3, y3));
    float h1 = dist(PointF(x0, y0), PointF(x3, y3));
    float h2 = dist(PointF(x1, y1), PointF(x2, y2));

    dst_w = static_cast<int>(std::max(w1, w2));
    dst_h = static_cast<int>(std::max(h1, h2));

    // Ensure valid dimensions
    dst_w = std::max(dst_w, 8);
    dst_h = std::max(dst_h, 8);

    // 3. Allocate destination buffer
    dst_data.resize(dst_w * dst_h * channels);
    
    // 4. Perspective Transform (Bilinear Interpolation)
    for (int y = 0; y < dst_h; y++) {
        for (int x = 0; x < dst_w; x++) {
            float u = (float)x / dst_w;
            float v = (float)y / dst_h;

            // Interpolate top and bottom edges
            float top_x = (1.0f - u) * x0 + u * x1;
            float top_y = (1.0f - u) * y0 + u * y1;
            
            float bottom_x = (1.0f - u) * x3 + u * x2;
            float bottom_y = (1.0f - u) * y3 + u * y2;

            // Interpolate vertically
            float src_x = (1.0f - v) * top_x + v * bottom_x;
            float src_y = (1.0f - v) * top_y + v * bottom_y;

            // Clamp and round to nearest integer
            int sx = static_cast<int>(std::round(src_x));
            int sy = static_cast<int>(std::round(src_y));
            
            // Clamp to source image boundaries
            sx = std::max(0, std::min(sx, src_w - 1));
            sy = std::max(0, std::min(sy, src_h - 1));

            // Copy pixel (all channels)
            int dst_idx = (y * dst_w + x) * channels;
            int src_idx = (sy * src_w + sx) * channels;

            for (int c = 0; c < channels; c++) {
                dst_data[dst_idx + c] = src_data[src_idx + c];
            }
        }
    }

    // 5. Handle Vertical Text (Result is taller than wide)
    if (dst_h > 1.5 * dst_w) {
        std::vector<uint8_t> rotated_data(dst_w * dst_h * channels);
        for (int y = 0; y < dst_h; y++) {
            for (int x = 0; x < dst_w; x++) {
                int src_idx = (y * dst_w + x) * channels;
                // Rotate 90 CW: (x, y) -> (h-1-y, x)
                int dst_x = dst_h - 1 - y;
                int dst_y = x;
                int dst_idx = (dst_y * dst_h + dst_x) * channels;
                
                for (int c = 0; c < channels; c++) {
                    rotated_data[dst_idx + c] = dst_data[src_idx + c];
                }
            }
        }
        dst_data = std::move(rotated_data);
        std::swap(dst_w, dst_h);
    }
}


//Function to rotate image 180 degrees
static void rotate180(const uint8_t* src_data, int w, int h, int c,
                      std::vector<uint8_t>& dst_data) {
    dst_data.resize(w * h * c);
    
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int src_idx = (y * w + x) * c;
            int dst_y = h - 1 - y;
            int dst_x = w - 1 - x;
            int dst_idx = (dst_y * w + dst_x) * c;
            
            for (int ch = 0; ch < c; ch++) {
                dst_data[dst_idx + ch] = src_data[src_idx + ch];
            }
        }
    }
}


// Helper function to get points from box coordinates
static std::vector<PointF> getBoxPoints(const float* pts) {
    std::vector<PointF> box_points(4);
    for (int i = 0; i < 4; i++) {
        box_points[i] = {pts[i * 2], pts[i * 2 + 1]};
    }
    return box_points;
}


// Helper function for debug visualization
static void draw_debug_boxes(const std::string& image_path,
                            const std::vector<TextBox>& boxes,
                            const std::string& output_path) {
    int img_w, img_h, img_channels;
    uint8_t* img_data = stbi_load(image_path.c_str(), &img_w, &img_h, &img_channels, 0);
    if (!img_data) {
        std::cerr << "Failed to load image for debug: " << image_path << "\n";
        return;
    }

    // Convert to RGBA for drawing
    std::vector<uint8_t> image_rgba(img_w * img_h * 4);
    if (img_channels == 4) {
        image_rgba.assign(img_data, img_data + (img_w * img_h * 4));
    } else if (img_channels == 3) {
        auto* src = img_data;
        auto* dst = image_rgba.data();
        for (int i = 0; i < img_w * img_h; i++) {
            dst[0] = src[0];
            dst[1] = src[1];
            dst[2] = src[2];
            dst[3] = 255;
            src += 3;
            dst += 4;
        }
    } else {
        auto* src = img_data;
        auto* dst = image_rgba.data();
        for (int i = 0; i < img_w * img_h; i++) {
            uint8_t val = *src++;
            dst[0] = val;
            dst[1] = val;
            dst[2] = val;
            dst[3] = 255;
            dst += 4;
        }
    }

    // Draw each box with a different color
    const uint8_t colors[][3] = {
        {255, 0, 0},    // Red
        {0, 255, 0},    // Green
        {0, 0, 255},    // Blue
        {255, 255, 0},  // Yellow
        {255, 0, 255},  // Magenta
        {0, 255, 255},  // Cyan
        {128, 0, 0},    // Dark Red
        {0, 128, 0},    // Dark Green
        {0, 0, 128},    // Dark Blue
        {128, 128, 0}   // Olive
    };

    auto draw_line = [&](int x1, int y1, int x2, int y2, const uint8_t color[3], int thickness = 2) {
        int dx = abs(x2 - x1);
        int dy = abs(y2 - y1);
        int sx = (x1 < x2) ? 1 : -1;
        int sy = (y1 < y2) ? 1 : -1;
        int err = dx - dy;

        while (true) {
            for (int ty = -thickness/2; ty <= thickness/2; ty++) {
                for (int tx = -thickness/2; tx <= thickness/2; tx++) {
                    int px = x1 + tx;
                    int py = y1 + ty;
                    if (px >= 0 && px < img_w && py >= 0 && py < img_h) {
                        int idx = (py * img_w + px) * 4;
                        image_rgba[idx + 0] = color[0];
                        image_rgba[idx + 1] = color[1];
                        image_rgba[idx + 2] = color[2];
                    }
                }
            }

            if (x1 == x2 && y1 == y2) break;
            int e2 = 2 * err;
            if (e2 > -dy) {
                err -= dy;
                x1 += sx;
            }
            if (e2 < dx) {
                err += dx;
                y1 += sy;
            }
        }
    };

    for (size_t i = 0; i < boxes.size(); i++) {
        const auto& box = boxes[i];
        const uint8_t* color = colors[i % 10];
        
        for (int j = 0; j < 4; j++) {
            int x1 = static_cast<int>(box.pts[j * 2]);
            int y1 = static_cast<int>(box.pts[j * 2 + 1]);
            int x2 = static_cast<int>(box.pts[((j + 1) % 4) * 2]);
            int y2 = static_cast<int>(box.pts[((j + 1) % 4) * 2 + 1]);
            
            draw_line(x1, y1, x2, y2, color, 3);
        }
        
        if (boxes.size() <= 100) {
            int center_x = 0, center_y = 0;
            for (int j = 0; j < 4; j++) {
                center_x += static_cast<int>(box.pts[j * 2]);
                center_y += static_cast<int>(box.pts[j * 2 + 1]);
            }
            center_x /= 4;
            center_y /= 4;
            
            std::string num = std::to_string(i + 1);
            uint8_t digit_patterns[10][5] = {
                {0x0E, 0x11, 0x11, 0x11, 0x0E},
                {0x04, 0x0C, 0x04, 0x04, 0x0E},
                {0x0E, 0x01, 0x0E, 0x10, 0x0F},
                {0x0E, 0x01, 0x06, 0x01, 0x0E},
                {0x02, 0x06, 0x0A, 0x0F, 0x02},
                {0x0F, 0x10, 0x0E, 0x01, 0x0E},
                {0x0E, 0x10, 0x0E, 0x11, 0x0E},
                {0x0F, 0x01, 0x02, 0x04, 0x04},
                {0x0E, 0x11, 0x0E, 0x11, 0x0E},
                {0x0E, 0x11, 0x0E, 0x01, 0x0E}
            };
            
            for (size_t c = 0; c < num.size(); c++) {
                int digit = num[c] - '0';
                int start_x = center_x + static_cast<int>(c) * 8 - (num.size() * 4);
                
                for (int py = 0; py < 7; py++) {
                    for (int px = 0; px < 5; px++) {
                        if (digit >= 0 && digit < 10 && 
                            (digit_patterns[digit][px] >> (6-py)) & 1) {
                            int draw_x = start_x + px;
                            int draw_y = center_y + py - 3;
                            if (draw_x >= 0 && draw_x < img_w && 
                                draw_y >= 0 && draw_y < img_h) {
                                int idx = (draw_y * img_w + draw_x) * 4;
                                image_rgba[idx + 0] = 255 - color[0];
                                image_rgba[idx + 1] = 255 - color[1];
                                image_rgba[idx + 2] = 255 - color[2];
                            }
                        }
                    }
                }
            }
        }
    }

    stbi_write_png(output_path.c_str(), img_w, img_h, 4, image_rgba.data(), img_w * 4);
    std::cout << "Debug image saved to: " << output_path << "\n";
    
    stbi_image_free(img_data);
}


TextDetector::TextDetector(const std::string& det_model_path)
    : model_path_(det_model_path),
      env_(ORT_LOGGING_LEVEL_WARNING, "dbnet") {
    
    Ort::SessionOptions opts_sess;
    opts_sess.SetIntraOpNumThreads(1);
    opts_sess.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

#ifdef _WIN32
    session_ = Ort::Session(env_, to_wstring(model_path_).c_str(), opts_sess);
#else
    session_ = Ort::Session(env_, model_path_.c_str(), opts_sess);
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
        throw std::runtime_error("Failed to get input/output names from detection model");
    }
    
    std::cout << "Detection model loaded: " << model_path_ << std::endl;
}


std::vector<CropResult> TextDetector::detect_and_crop(const std::string& image_path,
                                                     const DetectOptions& opt) {
    int img_w, img_h, img_channels;
    uint8_t* img_data = stbi_load(image_path.c_str(), &img_w, &img_h, &img_channels, 0);
    if (!img_data) {
        throw std::runtime_error("Failed to load image: " + image_path);
    }

    std::vector<uint8_t> original_image(img_data, img_data + (img_w * img_h * img_channels));

    std::vector<uint8_t> image_rgb;
    if (img_channels >= 3) {
        image_rgb.resize(img_w * img_h * 3);
        auto* src = img_data;
        auto* dst = image_rgb.data();
        for (int i = 0; i < img_w * img_h; i++) {
            dst[0] = src[0];
            dst[1] = src[1];
            dst[2] = src[2];
            src += img_channels;
            dst += 3;
        }
    } else {
        image_rgb.resize(img_w * img_h * 3);
        auto* src = img_data;
        auto* dst = image_rgb.data();
        for (int i = 0; i < img_w * img_h; i++) {
            uint8_t val = *src++;
            dst[0] = val;
            dst[1] = val;
            dst[2] = val;
            dst += 3;
        }
    }
    
    int target = opt.target_size;
    float scale = target / (float)std::max(img_w, img_h);

    int resized_w = int(img_w * scale);
    int resized_h = int(img_h * scale);

    auto round32 = [](int x) { return (x + 31) / 32 * 32; };
    int new_w = round32(resized_w);
    int new_h = round32(resized_h);

    std::vector<uint8_t> resized(new_h * new_w * 3, 0);
    std::vector<uint8_t> resized_temp(resized_w * resized_h * 3);

    stbir_resize_uint8_linear(
        image_rgb.data(), img_w, img_h, 0,
        resized_temp.data(), resized_w, resized_h, 0,
        STBIR_RGB);

    for (int y = 0; y < resized_h; y++) {
        for (int x = 0; x < resized_w; x++) {
            int src_idx = (y * resized_w + x) * 3;
            int dst_idx = (y * new_w + x) * 3;
            resized[dst_idx] = resized_temp[src_idx];
            resized[dst_idx + 1] = resized_temp[src_idx + 1];
            resized[dst_idx + 2] = resized_temp[src_idx + 2];
        }
    }

    float mean[3] = {0.485f, 0.456f, 0.406f};
    float stdv[3] = {0.229f, 0.224f, 0.225f};

    std::vector<float> input_blob(1 * 3 * new_h * new_w);
    int idx = 0;
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < new_h; y++) {
            for (int x = 0; x < new_w; x++) {
                int pixel_idx = (y * new_w + x) * 3 + (2 - c);
                float v = resized[pixel_idx] / 255.0f;
                input_blob[idx++] = (v - mean[c]) / stdv[c];
            }
        }
    }

    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<int64_t> shape = {1, 3, new_h, new_w};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem, input_blob.data(), input_blob.size(),
        shape.data(), shape.size());

    std::vector<Ort::Value> output_tensors;
    try {
        output_tensors = session_.Run(
            Ort::RunOptions{nullptr},
            input_names_.data(), &input_tensor, 1,
            output_names_.data(), output_names_.size());
    } catch (const std::exception& e) {
        std::cout << "Warning: Using dynamic names failed: " << e.what() << std::endl;
        std::cout << "Trying fallback names..." << std::endl;
        
        const char* fallback_input_names[] = {"x"};
        const char* fallback_output_names[] = {"sigmoid_0.tmp_0", "out", "output", "prob", "predict"};
        
        for (const char* out_name : fallback_output_names) {
            try {
                const char* test_output_names[] = {out_name};
                output_tensors = session_.Run(
                    Ort::RunOptions{nullptr},
                    fallback_input_names, &input_tensor, 1,
                    test_output_names, 1);
                std::cout << "Success with output name: " << out_name << std::endl;
                break;
            } catch (...) {
                continue;
            }
        }
        
        if (output_tensors.empty()) {
            stbi_image_free(img_data);
            throw std::runtime_error("Failed to run detection model with any output name");
        }
    }

    if (output_tensors.empty() || !output_tensors[0].IsTensor()) {
        stbi_image_free(img_data);
        throw std::runtime_error("No valid output tensor from detection model");
    }

    float* prob_data = output_tensors[0].GetTensorMutableData<float>();
    auto tensor_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    auto tensor_shape = tensor_info.GetShape();
    
    int prob_height, prob_width;
    if (tensor_shape.size() == 3) {
        prob_height = static_cast<int>(tensor_shape[1]);
        prob_width = static_cast<int>(tensor_shape[2]);
    } else if (tensor_shape.size() == 4) {
        prob_height = static_cast<int>(tensor_shape[2]);
        prob_width = static_cast<int>(tensor_shape[3]);
    } else {
        stbi_image_free(img_data);
        throw std::runtime_error("Unexpected output tensor shape from detection model");
    }

    if (prob_height != new_h || prob_width != new_w) {
        std::cout << "Warning: Probability map size (" << prob_width << "x" << prob_height 
                  << ") doesn't match input size (" << new_w << "x" << new_h << ")" << std::endl;
    }

    std::vector<float> prob(prob_data, prob_data + prob_height * prob_width);

    std::vector<uint8_t> bitmap(prob_height * prob_width, 0);
    for (int i = 0; i < prob_height * prob_width; i++) {
        bitmap[i] = prob[i] > opt.bin_thresh ? 255 : 0;
    }

    auto components = find_connected_components(bitmap, prob_width, prob_height, opt.min_component_area);

    std::vector<TextBox> boxes;
    for (auto& component : components) {
        if (component.size() < 3) continue;

        auto hull = convex_hull(component);
        if (hull.size() < 3) continue;

        float area = polygon_area(hull);
        if (area < 10) continue;

        float score = box_score_fast(prob, prob_width, prob_height, hull);
        if (score < opt.box_thresh) continue;

        auto polys = unclip_polygon(hull, opt.unclip_ratio);
        if (polys.empty()) continue;

        for (auto& poly : polys) {
            if (poly.size() < 3) continue;

            auto corners = min_area_rect(poly);
            if (corners.size() != 4) continue;

            TextBox tb;
            tb.score = score;

            float center_x = 0, center_y = 0;
            for (const auto& p : corners) {
                center_x += p.x;
                center_y += p.y;
            }
            center_x /= 4;
            center_y /= 4;

            std::vector<std::pair<float, int>> angles;
            for (int i = 0; i < 4; i++) {
                float angle = atan2(corners[i].y - center_y, corners[i].x - center_x);
                angles.emplace_back(angle, i);
            }
            std::sort(angles.begin(), angles.end());

            for (int i = 0; i < 4; i++) {
                int id = angles[i].second;
                tb.pts[i * 2] = corners[id].x / scale;
                tb.pts[i * 2 + 1] = corners[id].y / scale;
            }
            boxes.push_back(tb);
        }
    }

    if (opt.debug_boxes && !boxes.empty()) {
        draw_debug_boxes(image_path, boxes, opt.debug_output);
    }

    // Initialize rotation classifier if enabled
    std::unique_ptr<RotationCLSModel> cls_model;
    if (opt.use_rotation_cls && !opt.cls_model_path.empty()) {
        try {
            cls_model = std::make_unique<RotationCLSModel>(opt.cls_model_path);
            std::cout << "Rotation classification model loaded: " << opt.cls_model_path << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to load rotation classification model: " << e.what() << std::endl;
            std::cerr << "Rotation correction disabled." << std::endl;
            cls_model.reset();
        }
    }

    std::vector<CropResult> results;
    results.reserve(boxes.size());

    // Create cropped directory using modern C++17 filesystem
    if (opt.save_crops) {
        try {
            namespace fs = std::filesystem;
            fs::create_directories("cropped");
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to create cropped directory: " << e.what() << std::endl;
        }
    }

    for (size_t i = 0; i < boxes.size(); i++) {
        std::vector<PointF> box_points = getBoxPoints(boxes[i].pts);

        int cw = 0, ch = 0;
        std::vector<uint8_t> cropped;
        
        GetRotateCropImage(original_image.data(), img_w, img_h, img_channels, box_points, cropped, cw, ch);
        
        if (cropped.empty() || cw <= 0 || ch <= 0) continue;

        // Apply rotation correction if classifier is enabled
        if (cls_model) {
            try {
                int rotation = cls_model->run(cropped, cw, ch, img_channels);
                
                if (rotation == 180) {
                    // Rotate the cropped image 180 degrees
                    std::vector<uint8_t> rotated;
                    rotate180(cropped.data(), cw, ch, img_channels, rotated);
                    cropped = std::move(rotated);
                    
                    std::cout << "    Box " << (i + 1) << ": 180 rotation detected and corrected\n";
                }
            } catch (const std::exception& e) {
                std::cerr << "Warning: Rotation classification failed for box " << i + 1 
                          << ": " << e.what() << std::endl;
            }
        }

        CropResult cr;
        cr.box = boxes[i];
        cr.w = cw;
        cr.h = ch;
        cr.c = img_channels;
        cr.data = std::move(cropped);

        if (opt.save_crops) {
            std::string filename = "cropped/box_" + std::to_string(i) + ".png";
            stbi_write_png(filename.c_str(), cr.w, cr.h, cr.c, cr.data.data(), cr.w * cr.c);
        }

        results.push_back(std::move(cr));
    }

    stbi_image_free(img_data);
    return results;
}

