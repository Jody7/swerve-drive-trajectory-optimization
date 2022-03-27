from pyomo.environ import *
from pyomo.dae import *

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

traj_time = 5.0
traj_samples = 100
mass = 10.0
inertia = 1.0
wb_len = 1.0

m = ConcreteModel()
m.t = ContinuousSet(bounds=(0, traj_time))

m.x = Var(m.t) # inertial frame
m.y = Var(m.t)
m.phi = Var(m.t)

m.omega = DerivativeVar(m.phi) # angular velocity
m.omega_dot = DerivativeVar(m.omega)
m.x_dot = DerivativeVar(m.x)
m.y_dot = DerivativeVar(m.y)

m.v_x = Var(m.t) # vehichle frame
m.v_y = Var(m.t)
m.v_x_dot = DerivativeVar(m.v_x)
m.v_y_dot = DerivativeVar(m.v_y)


swerve_x_bounds = (0, 5)
swerve_y_bounds = (0, 0)
swerve_steer_speed = (-math.radians(360*2), math.radians(360*2))

# 4 swerve wheel modules
#m.F_FR_v = Var(m.t)
m.F_FR_x = Var(m.t, bounds=swerve_x_bounds)
m.F_FR_y = Var(m.t, bounds=swerve_y_bounds)
m.FR_theta = Var(m.t)
m.FR_theta_dot = DerivativeVar(m.FR_theta, bounds=swerve_steer_speed)

#m.F_FL_v = Var(m.t)
m.F_FL_x = Var(m.t, bounds=swerve_x_bounds)
m.F_FL_y = Var(m.t, bounds=swerve_y_bounds)
m.FL_theta = Var(m.t)
m.FL_theta_dot = DerivativeVar(m.FL_theta, bounds=swerve_steer_speed)

#m.F_BR_v = Var(m.t)
m.F_BR_x = Var(m.t, bounds=swerve_x_bounds)
m.F_BR_y = Var(m.t, bounds=swerve_y_bounds)
m.BR_theta = Var(m.t)
m.BR_theta_dot = DerivativeVar(m.BR_theta, bounds=swerve_steer_speed)

#m.F_BL_v = Var(m.t)
m.F_BL_x = Var(m.t, bounds=swerve_x_bounds)
m.F_BL_y = Var(m.t, bounds=swerve_y_bounds)
m.BL_theta = Var(m.t)
m.BL_theta_dot = DerivativeVar(m.BL_theta, bounds=swerve_steer_speed)


# constraint to transform and integrate vehichle-frame to inertial-frame
m.x_dot_ode = Constraint(m.t, rule=lambda m, t: 
	m.x_dot[t] == m.v_x[t] * cos(m.phi[t]) - m.v_y[t] * sin(m.phi[t])
)
m.y_dot_ode = Constraint(m.t, rule=lambda m, t: 
	m.y_dot[t] == m.v_x[t] * sin(m.phi[t]) + m.v_y[t] * cos(m.phi[t])
)

def v_force_x(t):
	return (m.F_FR_x[t]*cos(m.FR_theta[t])-m.F_FR_y[t]*sin(m.FR_theta[t])) + \
		(m.F_FL_x[t]*cos(m.FL_theta[t])-m.F_FL_y[t]*sin(m.FL_theta[t])) + \
		(m.F_BR_x[t]*cos(m.BR_theta[t])-m.F_BR_y[t]*sin(m.BR_theta[t])) + \
		(m.F_BL_x[t]*cos(m.BL_theta[t])-m.F_BL_y[t]*sin(m.BL_theta[t]))

def v_force_y(t):
	return (m.F_FR_y[t]*cos(m.FR_theta[t])+m.F_FR_y[t]*sin(m.FR_theta[t])) + \
		(m.F_FL_y[t]*cos(m.FL_theta[t])+m.F_FL_y[t]*sin(m.FL_theta[t])) + \
		(m.F_BR_y[t]*cos(m.BR_theta[t])+m.F_BR_y[t]*sin(m.BR_theta[t])) + \
		(m.F_BL_y[t]*cos(m.BL_theta[t])+m.F_BL_y[t]*sin(m.BL_theta[t]))

def v_rot_moment(t):
	return 0.5*wb_len * ((m.F_FR_x[t]*sin(m.FR_theta[t])+m.F_FL_x[t]*sin(m.FL_theta[t])+m.F_BR_x[t]*sin(m.BR_theta[t])+m.F_BL_x[t]*sin(m.BL_theta[t])) + \
		(m.F_FR_y[t]*cos(m.FR_theta[t])+m.F_FL_y[t]*cos(m.FL_theta[t])+m.F_BR_y[t]*cos(m.BR_theta[t])+m.F_BL_y[t]*cos(m.BL_theta[t])) + \
		(m.F_FR_x[t]*cos(m.FR_theta[t])+m.F_FL_x[t]*cos(m.FL_theta[t])+m.F_BR_x[t]*cos(m.BR_theta[t])+m.F_BL_x[t]*cos(m.BL_theta[t])) + \
		(m.F_FR_y[t]*sin(m.FR_theta[t])+m.F_FL_y[t]*sin(m.FL_theta[t])+m.F_BR_y[t]*sin(m.BR_theta[t])+m.F_BL_y[t]*sin(m.BL_theta[t]))
	)


m.v_x_dot_ode = Constraint(m.t, rule=lambda m, t: 
	m.v_x_dot[t] == 1/mass * (v_force_x(t) + mass*m.v_y[t]*m.omega[t])
)
m.v_y_dot_ode = Constraint(m.t, rule=lambda m, t: 
	m.v_y_dot[t] == 1/mass * (v_force_y(t) - mass*m.v_x[t]*m.omega[t])
)

m.omega_dot_ode = Constraint(m.t, rule=lambda m, t: 
	m.omega_dot[t] == 1/inertia * v_rot_moment(t)
)

m.pc = ConstraintList()
m.pc.add(m.x[0]==0)
m.pc.add(m.y[0]==0)
m.pc.add(m.x_dot[0]==0)
m.pc.add(m.y_dot[0]==0)
m.pc.add(m.phi[0]==0)
m.pc.add(m.omega[0]==0)

def cost_function(m, t):
	#return (m.x[t]-sin(t/4)*5)**2 + (m.y[t]-cos(t/4)*5)**2 + (m.omega[t]-math.radians(45))**2
	return (m.x_dot[t] - 2)**2 + (m.y_dot[t])**2

m.integral = Integral(m.t, wrt=m.t, rule=cost_function)
m.obj = Objective(expr=m.integral)

TransformationFactory('dae.finite_difference').apply_to(m, wrt=m.t, nfe=traj_samples)
SolverFactory('ipopt').solve(m, tee=False)

x_res = np.array([m.x[t]() for t in m.t]).tolist()
y_res = np.array([m.y[t]() for t in m.t]).tolist()
phi_res = np.array([m.phi[t]() for t in m.t]).tolist()

fr_x_res = np.array([m.F_FR_x[t]() for t in m.t]).tolist()
fr_y_res = np.array([m.F_FR_y[t]() for t in m.t]).tolist()

fr_theta_res = np.array([m.FR_theta[t]() for t in m.t]).tolist()
fl_theta_res = np.array([m.FL_theta[t]() for t in m.t]).tolist()
br_theta_res = np.array([m.BR_theta[t]() for t in m.t]).tolist()
bl_theta_res = np.array([m.BL_theta[t]() for t in m.t]).tolist()

fig, ax = plt.subplots(figsize=(10,10))

plt.xlim([-10, 10])
plt.ylim([-10, 10])

patch_len = 0.5
patch_wid = 0.1

fr_patch = ax.add_patch(Rectangle((0,0), patch_len, patch_wid))
fl_patch = ax.add_patch(Rectangle((0,0), patch_len, patch_wid))
br_patch = ax.add_patch(Rectangle((0,0), patch_len, patch_wid))
bl_patch = ax.add_patch(Rectangle((0,0), patch_len, patch_wid))

for i, t in enumerate(m.t):
	print(t, sin(t), cos(t))
	x = x_res[i]
	y = y_res[i]
	phi = phi_res[i]

	fr_patch_x=x+wb_len/2-patch_len/2
	fr_patch_y=y-wb_len/2-patch_wid/2
	fl_patch_x=x+wb_len/2-patch_len/2
	fl_patch_y=y+wb_len/2-patch_wid/2

	br_patch_x=x-wb_len/2-patch_len/2
	br_patch_y=y-wb_len/2-patch_wid/2
	bl_patch_x=x-wb_len/2-patch_len/2
	bl_patch_y=y+wb_len/2-patch_wid/2

	fr_theta = math.degrees(fr_theta_res[i])
	fl_theta = math.degrees(fr_theta_res[i])
	br_theta = math.degrees(br_theta_res[i])
	bl_theta = math.degrees(bl_theta_res[i])

	new_wb_patch = Rectangle((x-wb_len/2, y-wb_len/2), wb_len, wb_len, fill=None, transform=Affine2D().rotate_deg_around(x,y, math.degrees(phi))+ax.transData)

	fr_patch.set_transform(Affine2D().translate(tx=fr_patch_x,ty=fr_patch_y) + Affine2D().rotate_deg_around(fr_patch_x+patch_len/2,fr_patch_y+patch_wid/2, fr_theta) + Affine2D().rotate_deg_around(x,y, math.degrees(phi))+ax.transData)
	fl_patch.set_transform(Affine2D().translate(tx=fl_patch_x,ty=fl_patch_y) + Affine2D().rotate_deg_around(fl_patch_x+patch_len/2,fl_patch_y+patch_wid/2, fl_theta) + Affine2D().rotate_deg_around(x,y, math.degrees(phi))+ax.transData)
	br_patch.set_transform(Affine2D().translate(tx=br_patch_x,ty=br_patch_y) + Affine2D().rotate_deg_around(br_patch_x+patch_len/2,br_patch_y+patch_wid/2, br_theta) + Affine2D().rotate_deg_around(x,y, math.degrees(phi))+ax.transData)
	bl_patch.set_transform(Affine2D().translate(tx=bl_patch_x,ty=bl_patch_y) + Affine2D().rotate_deg_around(bl_patch_x+patch_len/2,bl_patch_y+patch_wid/2, bl_theta) + Affine2D().rotate_deg_around(x,y, math.degrees(phi))+ax.transData)

	ax.add_patch(new_wb_patch)
	plt.pause(traj_time / traj_samples)

plt.show()