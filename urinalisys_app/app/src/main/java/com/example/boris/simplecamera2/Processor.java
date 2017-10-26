package com.example.boris.simplecamera2;

import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Size;
import org.opencv.features2d.AKAZE;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.imgproc.Imgproc;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.List;
import java.lang.Math;

import static org.opencv.imgproc.Imgproc.CHAIN_APPROX_SIMPLE;
import static org.opencv.imgproc.Imgproc.RETR_LIST;
import static org.opencv.imgproc.Imgproc.RETR_TREE;
import static org.opencv.imgproc.Imgproc.cvtColor;
import static org.opencv.imgproc.Imgproc.pyrDown;

public class ProcessorUpd {

    Mat image_source = new Mat();
    Mat image_denoised = new Mat();
    Mat image_binary = new Mat();
    Mat image_morphed = new Mat();
    Mat warp_matrix = new Mat();

    Mat template = new Mat();
    Mat roi = new Mat();
    Mat strip = new Mat();

    private static final String PAG = "Processor";

    private final double scale = 5.0;
    private final double templateWidthMm = 0;
    private final double templateHeightMm = 0;
    private final double roiWidthMm = 0;
    private final double roiHeightMm = 0;
    private final double stripWidthMm = 0;
    private final double stripHeightMm = 0;
    private final double templateWidthPx = templateWidthMm * scale;
    private final double templateHeightPx = templateHeightMm * scale;
    private final double roiWidthPx = roiWidthMm * scale;
    private final double roiHeightPx = roiHeightMm * scale;
    private final double stripWidthPx = stripWidthMm * scale;
    private final double stripHeightPx = stripHeightMm * scale;

    private final AKAZE descriptor = AKAZE.create();
    private final DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);

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
     * Function that returns
     * @param x
     * @return x if odd, x+1 if not
     */
    public static int odd(int x){
        if((x % 2) == 0){
            x += 1;
        }
        return x;
    }

    /**
     * PyrDown image n times.
     * @param image source image
     * @param n times
     * @return shrinked copy of image
     */
    private Mat pyrDownNTimes(Mat image, int n){
        if (n <= 1){
            return image;
        }
        Mat shrinked = image.clone();
        for (int j = 0; j < n; j++){
            pyrDown(shrinked, shrinked);
        }
        return shrinked;
    }

    /**
     * Resize image such that larger side is of toSize scale
     * @param image source image
     * @param toSize size to scale
     * @return resized copy
     */
    private Mat downsampleToSize(Mat image, int toSize){
        int maxSize;
        if (image.width() > image.height())
            maxSize = image.width();
        else
            maxSize = image.height();
        int n = (int) Math.round(Math.log(maxSize) / Math.log(2));
        return pyrDownNTimes(image, n);
    }

    /**
     * Apply histogram equalization to image
     * @param image source image
     * @return contrasted copy
     */
    public Mat contrast(Mat image){
        Mat contrasted = new Mat();
        Imgproc.equalizeHist(image, contrasted);
        return contrasted;
    }

    /**
     * Apply median blurring to image
     * @param image source image
     * @param ksize size of kernel
     * @return denoised copy
     */
    public Mat denoise(Mat image, int ksize){
        Mat denoised = new Mat();
        Imgproc.medianBlur(image, denoised, ksize);
        return denoised;
    }

    /**
     * Apply adaptive binarization to image
     * @param image source image
     * @param blockSize size of block for adaptive methods
     * @param c constant for adaptive methods
     * @return binarized copy
     */
    public Mat binarize(Mat image, int blockSize, int c){
        Mat binarized = new Mat();
        Imgproc.adaptiveThreshold(image, binarized, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
                Imgproc.THRESH_BINARY, blockSize, c);
        return binarized;
    }

    /**
     * Apply morphology opening to image
     * @param image source image
     * @param ksize size of structuring element
     * @return morphologized copy
     */
    public Mat morpologyOpen(Mat image, int ksize){
        Mat morphologized = new Mat();
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(ksize, ksize));
        Imgproc.morphologyEx(image, morphologized, Imgproc.MORPH_OPEN, kernel);
        return morphologized;
    }

    /**
     * Apply morphology closing to image
     * @param image source image
     * @param ksize size of structuring element
     * @return morphologized copy
     */
    public Mat morpologyClose(Mat image, int ksize){
        Mat morphologized = new Mat();
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(ksize, ksize));
        Imgproc.morphologyEx(image, morphologized, Imgproc.MORPH_CLOSE, kernel);
        return morphologized;
    }

    /**
     * Get index of contour with max area
     * @param contours list of contours
     * @return maxId
     */
    public int getMaxAreaContourId(List<MatOfPoint> contours){
        if (contours.size() == 0){
            String message = "Empty countours in getMaxAreaContourId function";
            Log.e(PAG, message);
        }
        double maxArea = -1.;
        int maxId = -1;
        for (int j = 0; j < contours.size(); j++){
            double area = Imgproc.contourArea(contours.get(j));
            if (area > maxArea){
                maxArea = area;
                maxId = j;
            }
        }
        return maxId;
    };

    /**
     * Approximate contour with polygon using eps-accuracy
     * @param contour source contour
     * @param eps max distance from contour to approximate point
     * @param closed boolean for contour-closedness
     * @return approximation with poly
     */
    public MatOfPoint approximateContour(MatOfPoint contour, double eps, boolean closed) {
        MatOfPoint2f contour2f = new MatOfPoint2f();
        contour.convertTo(contour2f, CvType.CV_32F);

        MatOfPoint2f approximation2f = new MatOfPoint2f();
        Imgproc.approxPolyDP(contour2f, approximation2f, eps, closed);

        MatOfPoint approximation = new MatOfPoint();
        approximation2f.convertTo(approximation, CvType.CV_32S);

        return approximation;
    }

    private int[] getCorrectArrangement(Mat rect){
        // simple creation just not to worry about anything

        int [] arrangement = {-1, -1, -1, -1};
        float[] rect_col_values = {-1, -1, -1, -1};
        // filling arrangement array
        for(int i = 0; i < 4; i++) {
            rect_col_values[i] = (float) rect.get(i, 0)[0];
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
     * Check contour for area / elongation / ... criteria
     * @param contour source contour
     * @param epsArea min area to be "good"
     * @param epsRatio max elongation to be "good"
     * @return boolean of "goodness"
     */
    private boolean checkContour(MatOfPoint contour, double epsArea, double epsRatio){
        if (Imgproc.contourArea(contour) < epsArea)
            return false;
        Rect brect = Imgproc.boundingRect(contour);
        double ratio;
        if (brect.height > brect.width)
            ratio = brect.height / brect.width;
        else
            ratio = brect.width / brect.height;
        if (ratio > epsRatio)
            return false;
        return true;
    }

    /**
     * Calculate matrix for srcPts -> dstPts transformation
     * @param srcPts source points
     * @param dstPts destination points
     * @return h warp matrix
     */
    private Mat getWarpMatrix(MatOfPoint srcPts, MatOfPoint dstPts){
        MatOfPoint2f srcPts2f = new MatOfPoint2f();
        MatOfPoint2f dstPts2f = new MatOfPoint2f();
        srcPts.convertTo(srcPts2f, CvType.CV_32F);
        dstPts.convertTo(dstPts2f, CvType.CV_32F);
        return Calib3d.findHomography(srcPts2f, dstPts2f, Calib3d.RANSAC, 3.0);
    }

    /**
     * Apply perspective transformation to image using warp matrix
     * @param image source image
     * @param warpMatrix matrix for transformation
     * @param destSize size of output
     * @return warped copy of image
     */
    private Mat warp(Mat image, Mat warpMatrix, Size destSize){
        Mat warped = new Mat();
        Imgproc.warpPerspective(image, warped, warpMatrix, destSize);
        return warped;
    }

    /**
     Apply complete pipeline to input
     @param bitmapToProcess and then return
     @return Mat warped image
     */

    public Mat detectTemplate(Bitmap bitmapToProcess){

        // Preprocessing
        image_source = this.getMatFromBitmap(bitmapToProcess);
        image_denoised = this.denoise(image_source, 3);
        image_binary = this.binarize(image_denoised, odd(image_source.width() / 5), 0);
        image_morphed = this.morpologyOpen(image_binary, odd(image_source.width() / 250));

        // Finding template contour polygon
        List <MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(image_morphed, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);
        int maxId = this.getMaxAreaContourId(contours);
        MatOfPoint contour = contours.get(maxId);
        MatOfPoint polygon = new MatOfPoint();
        MatOfPoint polygon_eps;
        double[] epss = {1e-4, 5 * 1e-4, 1e-3, 5 * 1e-3, 1e-2};
        for (double eps : epss) {
            polygon_eps = this.approximateContour(contour, eps, true);
            if (polygon_eps.toArray().length == 4)
                polygon_eps.copyTo(polygon);
            else
                Log.v(PAG, "Found 4-vertices polygon, yay!");
        }
        if (polygon.toArray().length == 0)
            Log.v(PAG, "Not found 4-vertices polygon, shit!");
        
        int[] correctArrangement = this.getCorrectArrangement(polygon);
        ArrayList<Point> srcPtsList = new ArrayList<>();
        for (int j = 0; j < 4; j++)
            srcPtsList.add(polygon.toList().get(correctArrangement[j]));
        MatOfPoint srcPts = new MatOfPoint();
        srcPts.fromList(srcPtsList);

        List distPtsList = new ArrayList<>();
        distPtsList.add(0, new Point(0., 0.));
        distPtsList.add(1, new Point(this.templateWidthPx, 0.));
        distPtsList.add(2, new Point(this.templateWidthPx, this.templateHeightPx));
        distPtsList.add(3, new Point(0., this.templateHeightPx));
        MatOfPoint dstPts = new MatOfPoint();
        dstPts.fromList(distPtsList);

        // TODO: Keypoints assessment

        // Transformation
        warp_matrix = this.getWarpMatrix(srcPts, dstPts);
        template = this.warp(image_source, warp_matrix, new Size(this.templateWidthPx, this.templateHeightPx));
        return template;
    }

    // TODO
    public Mat detectROI(Mat template){
        return roi;
    }

    // TODO
    public Mat detectStrip(Mat template){
        return strip;
    }

}
