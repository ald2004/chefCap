#include "parser.h"
typedef struct ddx{
    int nboxes;
    detection* dd;
}ddx;
#ifdef __cplusplus
extern "C" {
#endif
    ddx unitest(char* cfg, char* weights, char* inputimg,int gpu_id) {
        printf(" Try to load cfg: %s, weights: %s \n", cfg, weights);
        //network* net = (network*)xcalloc(1, sizeof(network));
        //*net = parse_network_cfg_custom(cfg, batch, 1);
        network net = parse_network_cfg_custom(cfg, 1, 1);    // set batch=1
#ifdef GPU
        cuda_set_device(gpu_id);
#endif // GPU
        ddx ddx;
        if (weights) {
            load_weights(&net, weights);
            net.gpu_index = gpu_id;
        }
        else {
            printf("no weights");
            return ddx;
        }
        net.benchmark_layers = 0;
        fuse_conv_batchnorm(net);
        /*calculate_binary_weights(net);
        srand(2222222);*/
        layer l = net.layers[net.n - 1];
        printf("\n rand() is %d l.classes is %d \n", rand(), l.classes);
        fflush(stdout);

        

        /*for (int i = 0; i < net.n; ++i) {
            layer l = net.layers[i];
            if (l.type == YOLO) l.mean_alpha = 1.0 / 1;
        }*/
        //get_metadata();
        //do_nms_sort(dets, nboxes,6,0.45);
        /*for (int i = 0; i < nboxes; i++) {
            printf("\n c %d sortclass %d prob %f x %f y %f w %f h %f", 
                dets[i].classes, dets[i].sort_class,dets[i].prob ,dets[i].bbox.x, dets[i].bbox.y, dets[i].bbox.w, dets[i].bbox.h);
        }
        printf("\n");*/
        //free_detections(dets, nboxes);
        image in_img = load_image_color(inputimg, net.w, net.h);
        float* prediction = network_predict_image(&net, in_img);
        int nboxes = 0;
        
        ddx.dd = get_network_boxes(&net, net.w, net.h, 0.5, 0.5, NULL, 0, &nboxes, 0); // resized
        ddx.nboxes = nboxes;
        do_nms_sort(ddx.dd,nboxes, l.classes,0.45);
        return  ddx;
        
    }

    ddx detect_by_narray(char* cfg, char* weights, char* inputimg, int gpu_id) {
        printf(" Try to load cfg: %s, weights: %s \n", cfg, weights);
        //network* net = (network*)xcalloc(1, sizeof(network));
        //*net = parse_network_cfg_custom(cfg, batch, 1);
        network net = parse_network_cfg_custom(cfg, 1, 1);    // set batch=1
#ifdef GPU
        cuda_set_device(gpu_id);
#endif // GPU
        ddx ddx;
        if (weights) {
            load_weights(&net, weights);
            net.gpu_index = gpu_id;
        }
        else {
            printf("no weights");
            return ddx;
        }
        net.benchmark_layers = 0;
        fuse_conv_batchnorm(net);
        /*calculate_binary_weights(net);
        srand(2222222);*/
        layer l = net.layers[net.n - 1];
        printf("\n rand() is %d l.classes is %d \n", rand(), l.classes);
        fflush(stdout);
        image in_img = make_image(net.w,net.h,net.c);
        copy_image_from_bytes(in_img, inputimg);
        //printf(" \n %s \n",in_img);
        float* prediction = network_predict_image(&net, in_img);
        int nboxes = 0;

        ddx.dd = get_network_boxes(&net, net.w, net.h, 0.5, 0.5, NULL, 0, &nboxes, 0); // resized
        ddx.nboxes = nboxes;
        do_nms_sort(ddx.dd, nboxes, l.classes, 0.45);
        return  ddx;

    }

#ifdef __cplusplus
}
#endif // __cpluscplus

