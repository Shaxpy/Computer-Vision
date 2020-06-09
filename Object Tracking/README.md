## Optical Flow
### **Lucas-Kanade function**
Computes **Static** optical flow for a sparse feature set- typically the points we told to track.</br>
### Gunner Farneback algo
Computes **Dense** optical flow- To calculate flow of all points (It colors the video black if no movement is observed)
### **Meanshift Algo**
Identifies Clusters in subsequent iterations which may or may not be reasonable, (similar to K-means )- </br>
1.Given a target to track</br>
2.Compute color histogram and keep sliding the tracking window in iterations till the best match
### **CamShift Algo**
1.Meanshift happens</br>
2.New ROI is calculated</br>
3.The orientation of best fitting ellipse is then calculated and Mean shift is applied to the new scaled ROI 

### **High Speed Tracking**
For kernel regression, we derive a new Kernelized Correlation Filter (KCF), that unlike other kernel algorithms has the exact same complexity as its linear counterpart. Building on it, we also propose a fast multi-channel extension of linear correlation filters, via a linear kernel, which we call Dual Correlation Filter (DCF).

