package com.example.boris.simplecamera2;

import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

import static org.opencv.imgproc.Imgproc.CHAIN_APPROX_SIMPLE;
import static org.opencv.imgproc.Imgproc.RETR_TREE;


/**
 * Created by boris on 07.08.17.
 */

public class Processor {

    Mat image_source = new Mat();
    Mat image_binary = new Mat();
    Mat warp_matrix = new Mat();
    Mat image_warped = new Mat();

    private static final String PAG = "Processor";

    private final int height = 480;
    private final int stripes_num = 20;
    private final int stripes_size = 100;
    private final int d_bil_fil = 50;
    private final int sigma_x_bil_fil = 50;
    private final int sigma_c_bil_fil = 25;
    private final int thres_block = 100;
    private final int thres_c = 0;
    private final int[] target_rect = {0,0,25*24,25};

    private final double STRIPE_A = 5.0;
    private final double STRIPE_B = 120.;

    private final double GAP_WIDTH = 2.43;
    private final double POOL_WIDTH = 5.0;

    private final double LEFT_MARGIN = 30.2;
    private final double SIDE_MARGIN = 4.0;

    public Bitmap grayScaler(Bitmap bitmap){

        Mat img = getMatFromBitmap(bitmap);
        int width = img.width();
        int height = img.height();

        img = binarize(img);

        Bitmap bmp = Bitmap.createBitmap(width,height, Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(img,bmp);

        return bmp;
    };

    /**
     * takes as argument
     * @param image of stripped test stripe and
     * @return int[] rect - coordinates of 'white' reference
     * rectangle
     */
    private int[] get_reference_rect(Mat image){

        int height = image.height();
        int width = image.width();

        double k = height/STRIPE_A;

        int[] rect = {(int)SIDE_MARGIN,(int) SIDE_MARGIN,
                (int)((LEFT_MARGIN - SIDE_MARGIN)*k),
                (int)(height - SIDE_MARGIN*k)};

        return rect;
    }

    private final static int white_value = 255;

    //TODO rewrite pro.py
    /*
    private Mat adjust_white_balance_wrt_rgb(Mat image,int[] color_w){

        Mat image_wb = image.clone();

        List<Mat> channels = new ArrayList<Mat>();

        double r_coeff = ((1.*white_value)/color_w[0]);
        double g_coeff = ((1.*white_value)/color_w[1]);
        double b_coeff = ((1.*white_value)/color_w[2]);

        Core.split(image,channels);


    }*/
    /**
     * Function that returns
     * @param x
     * @return x if odd, x+1 if not
     */

    public static int odd(int x){
        if((x%2) ==0){
            x += 1;
        }
        return x;
    }

    /**
     * takes
     * @param bitmap as argument and
     * @return Mat img - image that is converted to Mat
     */

    private Mat getMatFromBitmap(Bitmap bitmap){
        Mat img = new Mat();
        Utils.bitmapToMat(bitmap,img);
        return img;
    }

    /**
     * Resize the input Mat
     * @param image according to input
     * @param size and then
     * @return Mat resized_image;
     */
    public static Mat resize(Mat image,int size){
        int height;
        int width;
        height = image.height();
        width = image.width();

        int dwidth = (int) (((1.*size)/height)*width);

        Mat resized_image = new Mat();
        Size dSize = new Size((double)dwidth,(double) size);
        Imgproc.resize(image,resized_image,dSize);
        return resized_image;
    }

    //TODO consider what is private and what is public here

    /**
     * get max_id -  index of the largest contour among List of
     * @param contours and return this
     * @return max_id
     */
    private int get_largest_contour_id(java.util.List<MatOfPoint> contours){
        if (contours.size()==0){
                String message = "Empty countours in get_largest_contour_id function";
                Log.e(PAG,message);
            }

        double max_area = -1.;
        int max_id = -1;

        for (int j = 0; j<contours.size(); j++){
            double area = Imgproc.contourArea(contours.get(j));

            if (area > max_area){
                max_area = area;
                max_id = j;
            }
        }

        return max_id;
    };


    /**
     * get rectangle vertices from
     * @param contour and then Mat
     * @return rect
     */
    private Mat get_rect(MatOfPoint2f contour){

        RotatedRect bounding_rot_rect = Imgproc.minAreaRect(contour);
        Mat rect = new Mat();
        Imgproc.boxPoints(bounding_rot_rect,rect);

        return rect;
    }


    private int[] get_correct_arrangement(Mat rect){
        // simple creation just not to worry about anything

        int [] arrangement = {-1,-1,-1,-1};
        float[] rect_col_values = {-1,-1,-1,-1};
        //filling arrangement array
        for(int i=0;i<4;i++) {
            rect_col_values[i] = (float)rect.get(i,0)[0];
        }


        int[] ixs = ArrayUtils.argsort(rect_col_values);

        if (rect.get(ixs[0],1)[0] > rect.get(ixs[1],1)[0]){
            arrangement[0] = ixs[1];
            arrangement[3] = ixs[0];
        }else{
            arrangement[0] = ixs[0];
            arrangement[3] = ixs[1];
        }

        if(rect.get(ixs[2],1)[0]>rect.get(ixs[3],1)[0]){
            arrangement[1] = ixs[3];
            arrangement[2] = ixs[2];
        }else{
            arrangement[1] = ixs[2];
            arrangement[2] = ixs[3];
        }

        return arrangement;
    }



    /**
     * make binary image from Mat
     * @param image and then return Mat
     * @return binary_image
     *
     * Now is working
     */
    private Mat binarize(Mat image){

        Mat image_gray = new Mat();
        int dims = image.dims();
        /*if(image.dims() == 3){*/ //TODO: Does .dims() work in OpenCV Java?
            Imgproc.cvtColor(image,image_gray,
                    Imgproc.COLOR_RGB2GRAY);
        /*}else{
            if(image.dims()==2){
                image_gray = image;
            }
            else{
                Log.e(PAG,"Weird shape of image");
            }
        }*/

        int height = image.height();
        int width = image.width();

        double coeff = (((double)height)/stripes_num)/stripes_size;

        int d = this.odd((int)(d_bil_fil*coeff));
        int sigma_x = this.odd((int)(sigma_x_bil_fil*coeff));
        int sigma_c = sigma_c_bil_fil;

        int block = this.odd((int)(thres_block*coeff));

        int c = thres_c;

        Mat image_blurred = new Mat();
        Imgproc.bilateralFilter(image_gray,image_blurred,d,sigma_c,sigma_x);

        Mat image_binary = new Mat();

        Imgproc.adaptiveThreshold(image_blurred,image_binary,255,
                Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,Imgproc.THRESH_BINARY,
                block,c);

        return image_binary;
    }

    /**
     * some violent homography using
     * @param src_rect and then returning
     * @return h - homography matrix
     */
    private Mat get_homography(Mat src_rect){
        List dist_list = new ArrayList<Point>();

        dist_list.add(0,new Point((double)target_rect[0],
                (double)target_rect[1]));
        dist_list.add(1,new Point((double)target_rect[2],
                (double)target_rect[1]));
        dist_list.add(2,new Point((double)target_rect[2],
                (double)target_rect[3]));
        dist_list.add(3,new Point((double)target_rect[0],
                (double)target_rect[3]));


        MatOfPoint2f dist_rect = new MatOfPoint2f();
        dist_rect.fromList(dist_list);

        int[] arrangement = get_correct_arrangement(src_rect);

        Mat src_rect_copy = src_rect.clone();
  
        for(int i=0;i<4;i++){
            src_rect_copy.put(i,0,
                    src_rect.get(arrangement[i],0));
            src_rect_copy.put(i,1,
                    src_rect.get(arrangement[i],1));
        }

        List src_list = new ArrayList<Point>();
        for(int i=0;i<4;i++){
            double x = src_rect_copy.get(i,0)[0];
            double y = src_rect_copy.get(i,1)[0];

            src_list.add(i,new Point(x,y));
        }


        MatOfPoint2f src_rect_2f = new MatOfPoint2f();
        src_rect_2f.fromList(src_list);

        Mat h = Calib3d.findHomography(src_rect_2f,dist_rect);

        return h;
    }

    private Mat get_warp_matrix(Mat image){
        //TODO consider about doing correct throwing of exception about gray scale image

        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        final Mat hierarchy = new Mat();
        Imgproc.findContours(image, contours, hierarchy,
                RETR_TREE,CHAIN_APPROX_SIMPLE);

        int max_id = get_largest_contour_id(contours);

        // type changes
        Point[] max_contour = contours.get(max_id).toArray();
        MatOfPoint2f max_contour_2f = new MatOfPoint2f();
        max_contour_2f.fromArray(max_contour);
        //
        Mat rect = get_rect(max_contour_2f);

        Mat h = get_homography(rect);

        return h;
    }

    /**
     * making some warp transformation to Mat
     * @param src image using Mat
     * @param warp_matrix and then Mat
     * @return dst image warpPerspective
     */
    private Mat warp(Mat src, Mat warp_matrix){
        Mat dst = new Mat();
        Size dSize = new Size(target_rect[2],target_rect[3]);
        Imgproc.warpPerspective(src,dst,warp_matrix,dSize);

        return dst;
    }

    /**
     Apply complete pipeline to input
     @param bitmapToProcess and then return
     @return Mat warped image
     */

    public Mat detect(Bitmap bitmapToProcess){

        image_source = this.getMatFromBitmap(bitmapToProcess);

        image_source = this.resize(image_source,this.height);

        image_binary = this.binarize(image_source);

        warp_matrix = this.get_warp_matrix(image_binary);

        Mat image_warped = this.warp(image_source, warp_matrix);

        return image_warped;
    }

}
