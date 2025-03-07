# initial import
import numpy as np
import pyopencl as cl
import matplotlib.pyplot as plt
import gratopy_with_ray_driven.gratopy_with_ray_driven as gratopy
import scipy





# Transform mapping (deforming) the unit-sphere onto the ellipse
def Transform_elipse (a,b,theta,y):
	# compute relevant directions
	vartheta = np.array([np.cos(theta),np.sin(theta)])
	vartheta_perp = np.array([np.sin(theta),-np.cos(theta)])
	
	# execute deformation
	x= 1/a * np.dot(y,vartheta) * vartheta + 1/b * np.dot(y,vartheta_perp) * vartheta_perp
	return x
	
# Transform mapping (deforming) the ellipse onto  the unit-sphere
def Transform_elipse_inverse (a,b,theta,x):
	# compute relevant directions
	vartheta = np.array([np.cos(theta),np.sin(theta)])
	vartheta_perp = np.array([np.sin(theta),-np.cos(theta)])
	
	# execute deformation				
	y= a * np.dot(x,vartheta) * vartheta + b * np.dot(x,vartheta_perp) * vartheta_perp
	return y


# Calculate the (explicit) Radon transform of the characteristic function on the unit sphere
def solution_circle(alpha,xi):
	
	if (xi**2<1):
		# solution (1-xi^2)^0.5 where it makes sense, 0 otherwise
		return np.sqrt(1-xi**2) 
	else:
		return 0
	
# Calculate the Radon transform of the characteristic function on the sphere 
def solution_elipse (a,b,theta, phi,s):
	# determine relevant directions
	phi+=np.pi/2
	varphi = np.array([np.cos(phi),np.sin(phi)])
	varphi_perp = np.array([np.sin(phi),-np.cos(phi)])
	
	# Employ deformation to access relevant quantities
	varalpha_perp = Transform_elipse_inverse(a,b,theta, varphi_perp)						
	norm_varalpha_perp= np.linalg.norm(varalpha_perp)
	varalpha_perp /= norm_varalpha_perp
	varalpha = np.array ([varalpha_perp[1],-varalpha_perp[0]])
	alpha = np.acos(varalpha[0])
	
	
	varphi = Transform_elipse_inverse(a,b,theta, varphi)
	xi = s * np.dot(varalpha,varphi)
	
	# The Radon transform can be traced back to the transform of a circle		
	return  solution_circle(alpha,xi)/norm_varalpha_perp*2


# Create the array corresponding to the entire Radon transform of an ellipse
def find_complete_solution(a,b,theta):		
	# Empty array
	solution = np.zeros([number_detectors , PS.n_angles])
	
	# vector with detector positions
	ramps=(np.arange((number_detectors))-number_detectors*0.5+0.5)/number_detectors*2
	# Loop through all sinogram pixels
	for phi_index in range(PS.n_angles):
		for s_index in range(number_detectors):
			phi = PS.angles[phi_index]
			s = ramps[s_index]		
			solution[s_index,phi_index]=solution_elipse (a,b,theta, phi,s)
			
	return solution


# Function calculates exact error (in an L^2(]-1,1[) and a discrete sum over all angles sense)
# between a piecewise constant function g_delta and the Radon transform of the characteristic 
# function concerning the ellipse
def precise_error(g_delta,a,b,theta):
	 
	# Initialize sums with zero
	sums = 0

	
	# calculate detector pixel centers
	ramps=(np.arange((number_detectors))-number_detectors*0.5+0.5)/number_detectors*2
	
	# Loop through all angles
	for phi_index in range(PS.n_angles):
		phi = PS.angles[phi_index]
		phi+=np.pi/2
		varphi = np.array([np.cos(phi),np.sin(phi)])
		varphi_perp = np.array([np.sin(phi),-np.cos(phi)])
		
		varalpha_perp = Transform_elipse_inverse(a,b,theta, varphi_perp)
		
		# Calculate relevant quantities for transformation to unit sphere
		eta = np.linalg.norm(varalpha_perp)
		varalpha_perp /= eta
		varalpha = np.array ([varalpha_perp[1],-varalpha_perp[0]])
		alpha = np.acos(varalpha[0])		
		varphi = Transform_elipse_inverse(a,b,theta, varphi)
		zeta = np.dot(varalpha,varphi)



		# Calculate the precise error along the detector line
		for s_index in range(PS.n_detectors):
			
			# Calculate the projection of (virtual) detector pixels on the transformed detector
			s=ramps[s_index]
			PP= max(-1,min(1,(s + PS.delta_s/2) * zeta))
			PM= max(-1,min(1,(s - PS.delta_s/2) * zeta))
			
			
			# Calculate the three parts of the norm
			term1 = g_delta[s_index,phi_index]**2 *PS.delta_s
			
			term2 = 1/eta**2 *1/zeta * 4* ((PP-PP**3/3)-(PM-PM**3/3))
			
			
			term3 = 1/eta/zeta  * ( PP*np.sqrt(1-PP**2) + np.asin(PP) 
									- PM*np.sqrt(1-PM**2) - np.asin(PM)  ) * g_delta[s_index,phi_index]
									

			# Summing up
			sums+=term1+term2-2*term3
	
	# Normalizing
	sums *= np.pi/PS.n_angles
	
	return np.sqrt(sums)
	




if __name__ == "__main__":
 
 
	# create pyopencl context
	ctx = cl.create_some_context()
	queue = cl.CommandQueue(ctx)


	# Everything will be executed in single precission and with Fortran contiguation
	dtype = np.dtype("float32")
	order = "F"


	#	Arrays to remember observed errors
	Error_naive_ray_total=[]
	Error_naive_pixel_total=[]
	
	Error_precise_ray_total=[]
	Error_precise_pixel_total=[]

 
 
	# discretization parameters
	
	#NX_interval = np.linspace(200,4000,20)
#	NX_interval = np.linspace(200,1000,11)
	#NX_interval = [200,1000,2000]
	#NX_interval = [4000]
	
	# List of resolutions that will be looped through. Can be adjusted to the users needs.
	NX_interval = np.linspace(200,4000,11)
	Angle_list = [180,360]
	
	print("\n \n Running code, dependent on your computer, this might take a few minutes. You might want to reduce the variable 'NX_interval' and Angle_ist if you are only interested in one specific resolution \n \n")
	print("Currently 'NX_interval'="+str(NX_interval)+"\n and 'Angle_list'="+str(Angle_list))

	for number_angles in Angle_list:
		Error_naive_ray = []
		Error_naive_pixel = []
		Error_precise_ray = []
		Error_precise_pixel = []
		

		for Nx in NX_interval:
			print("\n Currently running for Nx="+str(int(Nx))+" and number of angl="+str(number_angles)+"\n\n")
		
			# Use balanced resolutions
			Nx=int(Nx)
			number_detectors = Nx
			
			# create corresponding projectionsettings
			PS = gratopy.ProjectionSettings(queue, gratopy.RADON, Nx,
											number_angles, number_detectors)



			# Ellipse parameters
			elipse_a = 1
			elipse_b = 2
			elipse_theta = np.pi/4
			elipse_radius = 0.5
			
			elipse_a/=elipse_radius
			elipse_b/=elipse_radius

			# Create phantom relating to the ellipse			
			X=np.zeros([Nx,Nx])
			rampx= 2*(np.arange(Nx)-Nx*0.5+0.5)/Nx
			for x in range(Nx):
				X[:,x]=rampx
			Y=X.T
			
			phantom=np.zeros([Nx,Nx])
			phantom[np.where(  elipse_a**2 * (X*np.cos(elipse_theta)+Y*np.sin(elipse_theta))**2+  elipse_b**2 * (X*np.sin(elipse_theta)-Y*np.cos(elipse_theta))**2    <=1)]=1
			phantom= cl.array.to_device(queue, np.require(phantom,dtype,order))

			

			# Calculate the ray-driven and pixel-driven forward projections								
			sino_ray = gratopy.forwardprojection(phantom, PS,method=gratopy.RAY)
			sino_pixel = gratopy.forwardprojection(phantom, PS,method=gratopy.PIXEL)
			


		 
		 
			#Calculate the exact projection
			Exact_solution = find_complete_solution(elipse_a,elipse_b,elipse_theta)

			# Calculate the naive solution as the l^2 distance from exact to calculated projections 
			Error_naive_ray.append(np.linalg.norm(Exact_solution-sino_ray.get())/np.linalg.norm(Exact_solution))
			Error_naive_pixel.append(np.linalg.norm(Exact_solution-sino_pixel.get())/np.linalg.norm(Exact_solution))
			
			# Calculate the precise L^2 error in the detector dimension, and l^2 with respect to angles
			Error_precise_ray.append(precise_error(sino_ray.get(),elipse_a,elipse_b,elipse_theta))
			Error_precise_pixel.append(precise_error(sino_pixel.get(),elipse_a,elipse_b,elipse_theta))
			


		
		
			# Plot first instance
			if (Nx == NX_interval[1]) and (number_angles == Angle_list[1]):
				
				# Plot the phantom
				plt.figure()
				plt.title("phantom")
				plt.imshow(phantom.get());
				plt.colorbar()
				#tikzplotlib.save("forward/ellipse/phantom.tex")
			
				# Plot ray-driven projection
				plt.figure()
				plt.title("ray-driven")
				plt.imshow(sino_ray.get())
				plt.colorbar()
				#tikzplotlib.save("forward/ellipse/ray_driven.tex")

				
				# Plot pixel-driven projection
				plt.figure()
				plt.title("pixel-driven")
				plt.imshow(sino_pixel.get())
				plt.colorbar()
				#tikzplotlib.save("forward/ellipse/pixel_driven.tex")

				
				# Plot the exact solution
				plt.figure()
				plt.title("Exact")
				plt.imshow(Exact_solution)
				plt.colorbar()
				#tikzplotlib.save("forward/ellipse/Exact.tex")
				
				# Plot the difference between ray-driven and exact solution
				plt.figure()
				plt.title("Error ray-driven")
				plt.imshow(Exact_solution-sino_ray.get())
				plt.colorbar()
				#tikzplotlib.save("forward/ellipse/Error_ray.tex")

				# Plot the difference between ray-driven and exact solution
				plt.figure()
				plt.title("Error pixel-driven")	
				plt.imshow(Exact_solution-sino_pixel.get())
				plt.colorbar()
				# tikzplotlib.save("forward/ellipse/Error_pixel.tex")
				
				plt.figure()
				plt.title("Projection for \phi=45^circ")
				forty_five_degrees = list(PS.angles).index(np.pi/4)
				ramps=(np.arange((number_detectors))-number_detectors*0.5+0.5)/number_detectors*2
				plt.plot(ramps, Exact_solution[:,forty_five_degrees],"--", linewidth=10)
				plt.plot(ramps, sino_ray.get()[:,forty_five_degrees],linewidth=5)
				plt.plot(ramps, sino_pixel.get()[:,forty_five_degrees],alpha=0.5,linewidth=5)
				
				plt.legend(["Exact","Ray-driven","Pixel-driven"])
		
		
		# memorize computed errors
		Error_naive_ray_total.append(Error_naive_ray)
		Error_naive_pixel_total.append(Error_naive_pixel)

		Error_precise_ray_total.append(Error_precise_ray)
		Error_precise_pixel_total.append(Error_precise_pixel)


				
		
		
	#import tikzplotlib
	
	# Plot Naive errors against Nx
	plt.figure()
	Legend = []
	for i in range(len(Angle_list)):
		plt.plot(NX_interval,np.array(Error_naive_ray_total[i]))
		plt.plot(NX_interval, np.array(Error_naive_pixel_total[i]))
		Legend.append("ray-driven N_phi="+str(Angle_list[i]))
		Legend.append("pixel-driven N_phi="+str(Angle_list[i]))
		plt.legend(Legend)
	
	plt.title("Comparison of errors naive")	
	#tikzplotlib.save("forward/ellipse/naive_devolpment.tex")
	

	# Plot Naive errors against Nx	
	plt.figure()
	for i in range(len(Angle_list)):
		plt.plot(NX_interval,np.array(Error_precise_ray_total[i]))
		plt.plot(NX_interval,np.array(Error_precise_pixel_total[i]))
		Legend.append("ray-driven N_phi="+str(Angle_list[i]))
		Legend.append("pixel-driven N_phi="+str(Angle_list[i]))
		plt.legend(Legend)
	plt.title("Comparison of errors precise")

	#tikzplotlib.save("forward/ellipse/precise_devolpment.tex")
	plt.show()
	
	

		
		
		
		
		
		




