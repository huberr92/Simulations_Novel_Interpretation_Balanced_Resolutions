# initial import
import numpy as np
import pyopencl as cl
import matplotlib.pyplot as plt
import gratopy_with_ray_driven.gratopy_with_ray_driven as gratopy



# create pyopencl context
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

# discretization parameters
Discretization_parameter_list = [(1000,1000,90),(4000,4000,90),(1000,4000,90)]


#Array to memorize errors, where the first column is for ray-driven errors, the second for pixel-driven ones
My_Errors = np.zeros([len(Discretization_parameter_list),2])

# Fixxing maximal value to plot. This can lead to truncation but improves visibility
plotmax=0.1

# counter during iteration
counter=0

# Fix data type to single precission and Fortran contiguity
dtype = np.dtype("float32")
order = "F"


# Loop through all relevant discretization pairs
for (Nx,number_detectors,number_angles) in Discretization_parameter_list:


	print("Calculating for number of detectors "+str(number_detectors)+" number of angles "+str(number_angles))

	# Create gratopy projection setting
	PS = gratopy.ProjectionSettings(queue, gratopy.RADON, Nx,
									number_angles, number_detectors)


	# Create sinogram with constant values 1
	Sinogram=np.ones([number_detectors,number_angles])

		
	# Create ground-truth backprojection
	X=np.zeros([Nx,Nx])
	rampx= 4*(np.arange(Nx)-Nx*0.5+0.5)/Nx
	for x in range(Nx):
		X[:,x]=rampx
	ground_truth=np.ones([Nx,Nx])*np.pi
	mask=np.zeros([Nx,Nx])
	mask[np.where((X**2+X.T**2)<(4*(0.9)))]=1
	ground_truth *= mask
	
	
	# Calculate backprojections
	sino= cl.array.to_device(queue, np.require(Sinogram,dtype,order))
	backproj_ray = gratopy.backprojection(sino, PS,method=gratopy.RAY)
	backproj_pixel = gratopy.backprojection(sino, PS,method=gratopy.PIXEL)
	backproj_ray=backproj_ray.get()*mask
	backproj_pixel=backproj_pixel.get()*mask
	
	
	
	# Calculate errors (using that both functions are piecewise constant)
	True_Error=[]
	for approx in [backproj_ray,backproj_pixel]:
		
		## \|f-g\|**2 =\|f\|**2+\|g\|**2-2*<f,g> written as a pointwise sum to avoid rounding errors.
		pointwise_summands=(ground_truth**2)+approx**2-2*approx*ground_truth
		
		norm = np.sqrt(np.sum(pointwise_summands)*PS.delta_x**2)

		True_Error.append(norm)
	
	# Memorize relative errors
	My_Errors[counter,0]=True_Error[0]/np.linalg.norm(ground_truth)/PS.delta_x;
	My_Errors[counter,1]=True_Error[1]/np.linalg.norm(ground_truth)/PS.delta_x
	


	# Plotting results (Only plot of the ray-driven error is acticated
	
	# Plot ground truth
	if False:
		plt.figure()
		plt.title("ground_truth")
		plt.imshow(ground_truth, cmap="gray")
		plt.colorbar()


	# Plot the used sinogram
	if False:
		plt.figure()
		plt.title("Sinogram")
		plt.imshow(sino.get(), cmap="gray")
		plt.colorbar()

	# Plot the Ray-driven Backprojection
	if False:
		plt.figure()
		plt.title("Ray-driven Backprojection for Nx="+str(Nx)+" Ns="+str(number_detectors)+
					" number of angles ="+str(number_angles)+"\n"+"relative error = "+str(My_Errors[counter,0]))
		plt.imshow(backproj_ray, cmap="gray")
		plt.colorbar()


	# Plot the Pixel-driven Backprojection
	if False:
		plt.figure()
		plt.title("Pixel-driven Backprojection  for Nx="+str(Nx)+" Ns="+str(number_detectors)+
					" number of angles ="+str(number_angles)+"\n"+"relative error = "+str(My_Errors[counter,1]))
		plt.imshow(backproj_pixel, cmap="gray")
		plt.colorbar()


	# Plot the Error for the ray-driven backprojection
	if True:
		differences_ray=abs(ground_truth-backproj_ray)
		plt.figure()
		plt.title("Error in Ray-driven Backprojection for Nx="+str(Nx)+" Ns="+str(number_detectors)+
					" number of angles ="+str(number_angles)+"\n"+"relative error = "+str(My_Errors[counter,0]))
		plt.imshow(differences_ray,vmin=0,vmax=plotmax)
		error_ray=np.linalg.norm(differences_ray)
		plt.colorbar()


	# Plot the Error for the pixel-driven backprojection
	if False:
		differences_pixel=abs(ground_truth-backproj_pixel)
		plt.figure()
		plt.title("Difference pixel-driven for Nx="+str(Nx)+" Ns="+str(number_detectors)+
					" number of angles ="+str(number_angles)+"\n"+"relative error = "+str(My_Errors[counter,1]))
		plt.imshow(differences_pixel,vmin=0,vmax=plotmax)
		plt.colorbar()
		error_pixel=np.linalg.norm(differences_pixel)
		
	
	# Update counter
	counter+=1

counter-=1
plt.show()













