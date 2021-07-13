#ifndef _IMAGEDETECTOR_H_
#define _IMAGEDETECTOR_H_

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include "detector.hpp"

class imgdetector
{
    public:
    imgdetector(ros::NodeHandle& nh);
    ~imgdetector();

    void initDetector(std::string cfg_filename, std::string weight_filename, std::string class_filename, int gpu_id = 0);
    void imgcallback(const sensor_msgs::Image &msg);

    private:
    Detector *m_pdetector;
    std::vector<std::string> obj_names;
    image_transport::ImageTransport it;
    image_transport::Publisher img_pub;
    image_transport::Subscriber img_sub;
};

#endif
