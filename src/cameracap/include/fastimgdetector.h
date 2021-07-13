#ifndef _FASTIMAGEDETECTOR_H_
#define _FASTIMAGEDETECTOR_H_

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include "Yolo3Detection.h"
#include "objmsg/obj.h"
#include "objmsg/objArray.h"

class fasterimgdetector
{
    public:
    fasterimgdetector(ros::NodeHandle& nh);
    ~fasterimgdetector();
    
    void initDetector(std::string net, int n_classes, float conf_thresh);
    void imgcallback(const sensor_msgs::Image &msg);
    

    public:
    tk::dnn::Yolo3Detection *m_pdetector;
    image_transport::ImageTransport it;
    image_transport::Publisher img_pub;
    image_transport::Subscriber img_sub;
    ros::Publisher objmsg_pub;
};

#endif