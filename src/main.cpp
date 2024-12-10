#include <stdint.h>
#include <stdlib.h>

//alright - so buffer_streamer is a class that is daemon-registered to pull data up to a cap
//data integrity is done internally by chksum + repull data + friends
//a click on the bar == a spawn of video frame streamer 
//so we are expecting something like - user gives movies - we read movie_header -> extract the movie_resolution + # of images per second - set the RenderMachine hz appropriately - and set_video_streamer() to run in a daemon loop
//essentially, render_machine references a panel - and a video_streamer, 
//render_machine has an update() function - which will fetch the frame on the panel every time it is invoked - 
//if the update is successful - a timecounter will tick and the progress bar will move forward
//update() is deamon-subscriptible


class BufferStreamerInterface{

    public:

        virtual ~BufferStreamerInterface() noexcept = default;
        virtual auto is_end_of_stream() -> bool = 0;
        virtual auto next(size_t buf_sz) -> std::string = 0 ;
        virtual auto has_next() -> bool = 0; 
};

class VideoFrameStreamerInterface{

    public:

        virtual ~VideoFrameStreamerInterface() noexcept = default;
        virtual auto next_frame() -> std::unique_ptr<Image> = 0;
        virtual auto is_end_of_stream() -> bool = 0;
        virtual auto has_next() -> bool = 0;
};

class RenderMachineInterface{

    public:

        virtual ~RenderMachineInterface() noexcept = default;
        virtual void set_video_streamer(std::unique_ptr<VideoFrameStreamerInterface>) = 0;
        virtual void set_time(std::chrono::nanoseconds) = 0;
        virtual void start() = 0;
        virtual void stop() = 0;
        virtual void set_render_hz(double) = 0;
        virtual void set_dim(Dimension) = 0;
        virtual void set_pos() = 0;
};

int main(){

}