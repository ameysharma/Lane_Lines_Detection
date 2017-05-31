How the project was approached.

First Step:- The first step of approach was to import video from a file and play it by using OpenCV2.So, I imported the video using cv2.VideoCapture Command.
             The function video_read(): provides all the details of the method used to import video.

Second Step:- The Second step in the process was now to convert normal video into grayscale and then apply gaussian blur using
              
                  			  cv2.GaussianBlur(image name,kernel size,kernel size,0)
							  
				After applying gaussian blur to the frames now the next step was to find out edges in the image using cv2.Canny command.
				       
					          cv2.Canny(image name,lower value of threshold,higher value of threshold)
				
				After applying the filter the next step was to draw a line on the edges of lane.So we used pollyfill cv2.fillPoly
							
								cv2.Polyfill(image name,array of the points,masking color that has to be ignored)
				Here
				            array of points is 
                                          np.array(parameters) //converting frames into arrays by defining the points that  is  to included in the image//
										np.array(four points partameters, data type)
				Then 
				        Applying Hough Transformations using cv2.HoughLines
						
						lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),minimum line length, maximum line gap)
				Afer this step 
						
						Drawning the lines on the image using :-for loop
						 
						 for x in lines(values of lines):
							for variables(x1,y1,x2,y2) in lines:
							     cv2.line(parameters)//drawing lines on the blank
								 and then compbining both image using
								 cv2.addWeighted() 
				All this step is displayed in the function named image_processing():
				
Last Step:- The Last step was to display the video using cv2.imshow('result',image name that is to be shown)


								 
				
										
							
			
     

