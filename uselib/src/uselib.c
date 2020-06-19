#include "uselib.h"
#include "utils.h"
#include "parser.h"
#include "http_stream.h"
#include <stdio.h>

static int nboxes = 0;
image fetch_in_threadxx(char* in_f,int w,int h)
{
    return load_image_color(in_f, w, h);
}
detection* detect_in_threadxx(network net,image det_s,int letter_box)
{
    layer l = net.layers[net.n - 1];
    float* X = det_s.data;
    float* prediction = network_predict(net, X);

    if (letter_box)
        return get_network_boxes(&net, net.w,net.h, 0.5, 0.5, 0, 1, &nboxes, 1); // letter box
    else
        return  get_network_boxes(&net, net.w, net.h, 0.5, 0.5, 0, 1, &nboxes, 0); // resized
}
#ifdef __cplusplus
extern "C" {
#endif
    void unitest(char* cfg, char* weights, int clear, int batch,char* meta,char* inputimg,int gpu_id) {
        printf(" Try to load cfg: %s, weights: %s, clear = %d \n", cfg, weights, clear);
        //network* net = (network*)xcalloc(1, sizeof(network));
        //*net = parse_network_cfg_custom(cfg, batch, 1);
        network net = parse_network_cfg_custom(cfg, 1, 1);    // set batch=1
#ifdef GPU
        cuda_set_device(gpu_id);
#endif // GPU

        if (weights) {
            load_weights(&net, weights);
            net.gpu_index = gpu_id;
        }
        net.benchmark_layers = 0;
        fuse_conv_batchnorm(net);
        calculate_binary_weights(net);
        srand(2222222);
        layer l = net.layers[net.n - 1];
        printf("\n rand() is %d l.classes is %d ", rand(), l.classes);
        fflush(stdout);

        

        for (int i = 0; i < net.n; ++i) {
            layer l = net.layers[i];
            if (l.type == YOLO) l.mean_alpha = 1.0 / 1;
        }
        image in_img = fetch_in_threadxx(inputimg, net.w, net.h);
        detection* dets = detect_in_threadxx(net, in_img,0);
        for (int i = 0; i < nboxes; i++) {
            printf("\n c %d sortclass %d prob %f x %f y %f w %f h %f", dets[i].classes, dets[i].sort_class,dets[i].prob ,dets[i].bbox.x, dets[i].bbox.y, dets[i].bbox.w, dets[i].bbox.h);
        }
        printf("\n");
        free_detections(dets, nboxes);
        
    }

#ifdef __cplusplus
}
#endif // __cpluscplus
