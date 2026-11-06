#include "VT_Tracker.h"
#include <opencv2/core.hpp>
#include <cstring>
#include <memory>
#include <vector>

// C boundary for the Windows x64 H20 planar profile. Frames are contiguous B planes.
namespace {
struct State {
    VT_Tracker tracker;
    std::vector<cv::Point2f> grid, deformed, uv;
    cv::Mat reference;
};
}
extern "C" __declspec(dllexport) void* seed17_open(const unsigned char* reference) noexcept {
    try {
        if (!reference) return nullptr;
        cv::setNumThreads(1);
        auto state = std::make_unique<State>();
        int rows = 0, cols = 0;
        const std::vector<cv::Point2f> roi{{180,14},{628,81},{627,406},{176,467}};
        if (state->tracker.getMarkerPoints(roi,state->grid,rows,cols,5) != 69*82 ||
            rows != 69 || cols != 82) return nullptr;
        state->reference=cv::Mat(480,640,CV_8U,const_cast<unsigned char*>(reference)).clone();
        return state.release();
    } catch (...) {return nullptr;}
}
extern "C" __declspec(dllexport) int seed17_track(void* handle,const unsigned char* blue,
                                                  float* output, size_t capacity) noexcept {
    try {
        if (!handle || !blue || !output || capacity != 69*82*2) return -1;
        auto& s=*static_cast<State*>(handle);
        cv::Mat current(480,640,CV_8U,const_cast<unsigned char*>(blue));
        if (s.tracker.track(s.reference,s.grid,current,s.deformed,s.uv) != 0 ||
            s.uv.size()!=69*82) return -2;
        static_assert(sizeof(cv::Point2f)==2*sizeof(float));
        std::memcpy(output,s.uv.data(),capacity*sizeof(float));
        return 0;
    } catch (...) {return -3;}
}
extern "C" __declspec(dllexport) void seed17_close(void* handle) noexcept {
    delete static_cast<State*>(handle);
}
