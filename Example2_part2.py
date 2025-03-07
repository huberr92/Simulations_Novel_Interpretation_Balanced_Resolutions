# initial import
import numpy as np
import pyopencl as cl
import matplotlib.pyplot as plt
import gratopy_with_ray_driven.gratopy_with_ray_driven as gratopy


# create pyopencl context
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)


# We use single precision and Fortran contiguity
dtype = np.dtype("float32")
order = "F"



# discretization parameters
counter_angles=0	
Nx = 2000
List_Angles=np.linspace(30,720,47)
number_detectors=2000

# Empty array to memorize errors
My_Errors_single_angle = np.zeros([len(List_Angles),2])
My_Errors_constant_one = np.zeros([len(List_Angles),2])


# Create ground_truth
X=np.zeros([Nx,Nx])
rampx= 4*(np.arange(Nx)-Nx*0.5+0.5)/Nx
for x in range(Nx):
	X[:,x]=rampx
ground_truth_single_angle = np.ones([Nx,Nx])
ground_truth_constant_one = np.ones([Nx,Nx])*np.pi

mask=np.zeros([Nx,Nx])
mask[np.where((X**2+X.T**2)<(4*(0.9)))]=1
ground_truth_single_angle *= mask
ground_truth_constant_one *= mask


print("\n \n Running code, dependent on your computer, this might take a few minutes. You might want to reduce the variable 'NX_interval' and Angle_ist if you are only interested in one specific resolution \n \n")
print("Currently 'Nx'="+str(Nx)+"\n and 'number_detectors'="+str(number_detectors)+ "\n and List_Angles="+str(List_Angles)+"\n\n")


# Loop through all angles
for number_angles in List_Angles:
	number_angles=int(number_angles)
	#Foldername=experiment_name+"/"+str(number_angles)+"/"
	#try:
	#	os.makedirs(Foldername)
	#except OSError as error:
	#	print(error)    

	print("\nCurrently using number_angles=" + str(number_angles)+"\n")




	# Create Gratopy Projection Setting
	PS = gratopy.ProjectionSettings(queue, gratopy.RADON, Nx,
									number_angles, number_detectors)



	# Create sinogram that is constantly one
	Sinogram_constant_one = np.ones([number_detectors,number_angles])
	
	
	# Create sinogram with a single angle being active
	Sinogram_single_angle=np.zeros([number_detectors,number_angles])
	active_angles=[int(number_angles/4)]
	Sinogram_single_angle[:,active_angles] = 1/(np.pi*len(active_angles)/number_angles)
	
	
	
	# Calculate backprojections 
	sino_single_angle = cl.array.to_device(queue, np.require(Sinogram_single_angle,dtype,order))
	sino_constant_one = cl.array.to_device(queue, np.require(Sinogram_constant_one,dtype,order))
	backproj_ray_single_angle = gratopy.backprojection(sino_single_angle, PS,method=gratopy.RAY)
	backproj_pixel_single_angle = gratopy.backprojection(sino_single_angle, PS,method=gratopy.PIXEL)
	backproj_ray_constant_one = gratopy.backprojection(sino_constant_one, PS,method=gratopy.RAY)
	backproj_pixel_constant_one = gratopy.backprojection(sino_constant_one, PS,method=gratopy.PIXEL)
	backproj_ray_single_angle=backproj_ray_single_angle.get()*mask
	backproj_pixel_single_angle=backproj_pixel_single_angle.get()*mask	
	backproj_ray_constant_one=backproj_ray_constant_one.get()*mask
	backproj_pixel_constant_one=backproj_pixel_constant_one.get()*mask

	
	# Calculate errors for single angle sinogram
	True_Error_single_angle=[]
	for approx in [backproj_ray_single_angle,backproj_ray_single_angle]:
		
		pointwise_summands=(ground_truth_single_angle**2)+approx**2-2*approx*ground_truth_single_angle
		norm = np.sqrt(np.sum(pointwise_summands)*PS.delta_x**2) /np.linalg.norm(ground_truth_single_angle)/PS.delta_x
		True_Error_single_angle.append(norm)

	# Calculate errors for the constantly one sinogram
	True_Error_constant_one=[]
	for approx in [backproj_ray_constant_one,backproj_pixel_constant_one]:
		
		pointwise_summands=(ground_truth_constant_one**2)+approx**2-2*approx*ground_truth_constant_one
		norm = np.sqrt(np.sum(pointwise_summands)*PS.delta_x**2) /np.linalg.norm(ground_truth_constant_one)/PS.delta_x
		True_Error_constant_one.append(norm)
		

	# Memorize errors
	My_Errors_single_angle[counter_angles,0]=True_Error_single_angle[0];My_Errors_single_angle[counter_angles,1]=True_Error_single_angle[1]
	My_Errors_constant_one[counter_angles,0]=True_Error_constant_one[0];My_Errors_constant_one[counter_angles,1]=True_Error_constant_one[1]

	counter_angles+=1



# Plot Errors for ray-driven methods with increasing number of angles
plt.figure()
plt.plot(List_Angles,My_Errors_single_angle) 
plt.plot(List_Angles,My_Errors_constant_one)
plt.title("Error with increasing number of angles") 
plt.xlabel("Number of angles")
plt.ylabel("L2 error")
plt.legend(["rd for a single angle sinogram", "rd for constant sinogram" ])
#tikzplotlib.save(experiment_name+"/development.tex")
plt.show()
