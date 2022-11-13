
#include <opencv2/opencv.hpp>
#include <algorithm>

#include <unordered_map>
#include <memory>
#include "pybind11.hpp"
#include "ffhdd/ffmpeg-demuxer.hpp"
#include "ffhdd/cuvid-decoder.hpp"

using namespace std;
using namespace cv;

namespace py=pybind11;

class FFmpegDemuxer { 
public:
	FFmpegDemuxer(std::string uri, bool auto_reboot = false){

		instance_ = FFHDDemuxer::create_ffmpeg_demuxer(
			uri, 
			auto_reboot
		);
	}

	bool valid(){
		return instance_ != nullptr;
	}

	FFHDDemuxer::IAVCodecID get_video_codec() {return instance_->get_video_codec();}
	virtual FFHDDemuxer::IAVPixelFormat get_chroma_format(){return instance_->get_chroma_format();}
	virtual int get_width() {return instance_->get_width();}
	virtual int get_height() {return instance_->get_height();}
	virtual int get_bit_depth() {return instance_->get_bit_depth();}
	virtual int get_fps() {return instance_->get_fps();}
	virtual int get_total_frames() {return instance_->get_total_frames();}
	virtual py::tuple get_extra_data() {
		uint8_t* pdata = nullptr;
		int pbytes = 0;
		instance_->get_extra_data(&pdata, &pbytes);
		return py::make_tuple((uint64_t)pdata, pbytes);
	}

	virtual bool isreboot() {return instance_->isreboot();}
	virtual void reset_reboot_flag() {instance_->reset_reboot_flag();}
	virtual py::tuple demux() {

		uint8_t* pdata = nullptr;
		int pbytes = 0;
		bool iskey = false;
		bool ok = instance_->demux(&pdata, &pbytes, &time_pts_, &iskey);
		return py::make_tuple((uint64_t)pdata, pbytes, time_pts_, iskey, ok);
	}

	virtual bool reopen() {return instance_->reopen();}

private:
	uint64_t time_pts_ = 0;
	shared_ptr<FFHDDemuxer::FFmpegDemuxer> instance_;
}; 

class CUVIDDecoder { 
public:
	CUVIDDecoder(bool bUseDeviceFrame, FFHDDemuxer::IAVCodecID eCodec, int max_cache, int gpu_id,
        int cl, int ct, int cr, int cb, int rw, int rh, FFHDDecoder::FrameType frametype){
		
		FFHDDecoder::IcudaVideoCodec codec = FFHDDecoder::ffmpeg2NvCodecId(eCodec);
		FFHDDecoder::CropRect crop{0, 0, 0, 0};
		FFHDDecoder::ResizeDim resize{0, 0};
		if(cr - cl > 0 && cb - ct > 0){
			crop.l = cl;
			crop.t = ct;
			crop.r = cr;
			crop.b = cb;
		}

		if(rw > 0 && rh > 0){
			resize.w = rw;
			resize.h = rh;
		}

		output_frametype_ = frametype;
		instance_ = FFHDDecoder::create_cuvid_decoder(
			bUseDeviceFrame, codec, max_cache, gpu_id, &crop, &resize, frametype
		);
	}

	bool valid(){
		return instance_ != nullptr;
	}

	py::tuple get_numpy(){
		uint64_t pts = 0;
		unsigned int frame_index;
		auto ptr = instance_->get_frame(&pts, &frame_index);
		if(output_frametype_ == FFHDDecoder::FrameType::YUV_NV12){
			int align_width = (instance_->get_width() + 15) / 16 * 16;
			auto image = py::array(py::dtype::of<unsigned char>(),
				vector<int>{int(instance_->get_height() * 1.5f), align_width}, ptr
			);
			return py::make_tuple(image, pts, frame_index);
		}else{
			auto image = py::array(py::dtype::of<unsigned char>(),
				vector<int>{instance_->get_height(), instance_->get_width(), 3}, ptr
			);
			return py::make_tuple(image, pts, frame_index);
		}
	}

	int get_frame_size() {return instance_->get_frame_size();}
	int get_width() {return instance_->get_width();}
	int get_height() {return instance_->get_height();}
	unsigned int get_frame_index() {return instance_->get_frame_index();}
	unsigned int get_num_decoded_frame() {return instance_->get_num_decoded_frame();}
	py::tuple get_frame() {
		uint64_t pts = 0;
		unsigned int frame_index;
		auto ptr = instance_->get_frame(&pts, &frame_index);
		return py::make_tuple((uint64_t)ptr, pts, frame_index);
	}
	int decode(uint64_t pData, int nSize, int64_t nTimestamp=0) {
		const uint8_t* ptr = (const uint8_t*)pData;
		return instance_->decode(ptr, nSize, nTimestamp);
	}
	int64_t get_stream() {return (uint64_t)instance_->get_stream();}

private:
	shared_ptr<FFHDDecoder::CUVIDDecoder> instance_;
	FFHDDecoder::FrameType output_frametype_;
}; 

int app_multi_camera();
int main(){
	app_multi_camera();
	return 0;
}

PYBIND11_MODULE(libffhdd, m){

	py::enum_<FFHDDecoder::FrameType>(m, "FrameType")
		.value("Unknow",   FFHDDecoder::FrameType::Unknow)
		.value("BGR", FFHDDecoder::FrameType::BGR)
		.value("RGB",   FFHDDecoder::FrameType::RGB)
		.value("YUV_NV12",   FFHDDecoder::FrameType::YUV_NV12);

	py::class_<FFmpegDemuxer>(m, "FFmpegDemuxer")
		.def(py::init<string, bool>(), py::arg("uri"), py::arg("auto_reboot")=false)
		.def_property_readonly("valid", &FFmpegDemuxer::valid)
		.def("get_video_codec", &FFmpegDemuxer::get_video_codec)
		.def("get_chroma_format", &FFmpegDemuxer::get_chroma_format)
		.def("get_width", &FFmpegDemuxer::get_width)
		.def("get_height", &FFmpegDemuxer::get_height)
		.def("get_bit_depth", &FFmpegDemuxer::get_bit_depth)
		.def("get_fps", &FFmpegDemuxer::get_fps)
		.def("get_total_frames", &FFmpegDemuxer::get_total_frames)
		.def("get_extra_data", &FFmpegDemuxer::get_extra_data)
		.def("isreboot", &FFmpegDemuxer::isreboot)
		.def("reset_reboot_flag", &FFmpegDemuxer::reset_reboot_flag)
		.def("demux", &FFmpegDemuxer::demux)
		.def("reopen", &FFmpegDemuxer::reopen);

	py::class_<CUVIDDecoder>(m, "CUVIDDecoder")
		.def(py::init<bool, FFHDDemuxer::IAVCodecID, int, int, int, int, int, int, int, int, FFHDDecoder::FrameType>(), 
			py::arg("bUseDeviceFrame"), py::arg("codec"), py::arg("max_cache"), py::arg("gpu_id"),
			py::arg("crop_left")=0, py::arg("crop_top")=0, py::arg("crop_right")=0, py::arg("crop_bottom")=0,
			py::arg("resize_width")=0, py::arg("resize_height")=0,
			py::arg("output_type")=FFHDDecoder::FrameType::BGR
		)
		.def_property_readonly("valid", &CUVIDDecoder::valid)
		.def("get_numpy", &CUVIDDecoder::get_numpy)
		.def("get_frame_size", &CUVIDDecoder::get_frame_size)
		.def("get_width", &CUVIDDecoder::get_width)
		.def("get_height", &CUVIDDecoder::get_height)
		.def("get_frame_index", &CUVIDDecoder::get_frame_index)
		.def("get_num_decoded_frame", &CUVIDDecoder::get_num_decoded_frame)
		.def("get_frame", &CUVIDDecoder::get_frame)
		.def("decode", &CUVIDDecoder::decode)
		.def("get_stream", &CUVIDDecoder::get_stream);
};