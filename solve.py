from pyomo.environ import *
from pyomo.dae import *

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

traj_time = 15.0
traj_samples = traj_time / 0.05
mass = 5.0
inertia = 1.0
wb_len = 1.0

B_f = 1
C_f = 2
D_f = 0.3

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


swerve_x_bounds = (-10, 10)
swerve_y_bounds = (-999, 999)
swerve_steer_speed = (-math.radians(360*4), math.radians(360*4))

# 4 swerve wheel modules
m.FR_sa = Var(m.t)
m.F_FR_x = Var(m.t, bounds=swerve_x_bounds)
m.F_FR_y = Var(m.t, bounds=swerve_y_bounds)
m.FR_theta = Var(m.t)
m.FR_theta_dot = DerivativeVar(m.FR_theta, bounds=swerve_steer_speed)

m.FL_sa = Var(m.t)
m.F_FL_x = Var(m.t, bounds=swerve_x_bounds)
m.F_FL_y = Var(m.t, bounds=swerve_y_bounds)
m.FL_theta = Var(m.t)
m.FL_theta_dot = DerivativeVar(m.FL_theta, bounds=swerve_steer_speed)

m.BR_sa = Var(m.t)
m.F_BR_x = Var(m.t, bounds=swerve_x_bounds)
m.F_BR_y = Var(m.t, bounds=swerve_y_bounds)
m.BR_theta = Var(m.t)
m.BR_theta_dot = DerivativeVar(m.BR_theta, bounds=swerve_steer_speed)

m.BL_sa = Var(m.t)
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
	return (m.F_FR_y[t]*cos(m.FR_theta[t])+m.F_FR_x[t]*sin(m.FR_theta[t])) + \
		(m.F_FL_y[t]*cos(m.FL_theta[t])+m.F_FL_x[t]*sin(m.FL_theta[t])) + \
		(m.F_BR_y[t]*cos(m.BR_theta[t])+m.F_BR_x[t]*sin(m.BR_theta[t])) + \
		(m.F_BL_y[t]*cos(m.BL_theta[t])+m.F_BL_x[t]*sin(m.BL_theta[t]))

def v_rot_moment(t):
	return 0.5*wb_len * \
		((m.F_FR_x[t]*sin(m.FR_theta[t]) + -m.F_FL_x[t]*sin(m.FL_theta[t]) + m.F_BR_x[t]*sin(m.BR_theta[t]) + -m.F_BL_x[t]*sin(m.BL_theta[t])) + \
		 (m.F_FR_y[t]*sin(m.FR_theta[t]) + -m.F_FL_y[t]*sin(m.FL_theta[t]) + m.F_BR_y[t]*sin(m.BR_theta[t]) + -m.F_BL_y[t]*sin(m.BL_theta[t])) + \
		 (m.F_FR_x[t]*cos(m.FR_theta[t]) + m.F_FL_x[t]*cos(m.FL_theta[t]) + -m.F_BR_x[t]*cos(m.BR_theta[t]) + -m.F_BL_x[t]*cos(m.BL_theta[t])) + \
		 (m.F_FR_y[t]*cos(m.FR_theta[t]) + m.F_FL_y[t]*cos(m.FL_theta[t]) + -m.F_BR_y[t]*cos(m.BR_theta[t]) + -m.F_BL_y[t]*cos(m.BL_theta[t]))
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

def solv_atan(y, x):
	return Expr_if(x**2 <= 0.1, atan(y), atan(y / x))
def solv_scal(v_x, v_y):
	mag = v_x**2+v_y**2
	return atan(mag) / (math.pi*0.5)
	#return Expr_if(mag <= 0.1, mag, atan(mag) / (math.pi*0.5))

m.FR_sa_ode = Constraint(m.t, rule=lambda m, t: 
	m.FR_sa[t] == solv_scal(m.v_x[t], m.v_y[t])*(solv_atan((m.v_y[t]+0.5*wb_len*m.omega[t]), (m.v_x[t]-0.5*wb_len*m.omega[t])) - m.FR_theta[t])
)
m.FL_sa_ode = Constraint(m.t, rule=lambda m, t: 
	m.FL_sa[t] == solv_scal(m.v_x[t], m.v_y[t])*(solv_atan((m.v_y[t]+0.5*wb_len*m.omega[t]), (m.v_x[t]+0.5*wb_len*m.omega[t])) - m.FL_theta[t])
)
m.BR_sa_ode = Constraint(m.t, rule=lambda m, t: 
	m.BR_sa[t] == solv_scal(m.v_x[t], m.v_y[t])*(solv_atan((m.v_y[t]-0.5*wb_len*m.omega[t]), (m.v_x[t]-0.5*wb_len*m.omega[t])) - m.BR_theta[t])
)
m.BL_sa_ode = Constraint(m.t, rule=lambda m, t: 
	m.BL_sa[t] == solv_scal(m.v_x[t], m.v_y[t])*(solv_atan((m.v_y[t]-0.5*wb_len*m.omega[t]), (m.v_x[t]+0.5*wb_len*m.omega[t])) - m.BL_theta[t])
)

m.F_FR_y_ode = Constraint(m.t, rule=lambda m, t: 
	m.F_FR_y[t] == -D_f*sin(C_f*atan(B_f*m.FR_sa[t]))*mass*9.8
)
m.F_FL_y_ode = Constraint(m.t, rule=lambda m, t: 
	m.F_FL_y[t] == -D_f*sin(C_f*atan(B_f*m.FL_sa[t]))*mass*9.8
)
m.F_BR_y_ode = Constraint(m.t, rule=lambda m, t: 
	m.F_BR_y[t] == -D_f*sin(C_f*atan(B_f*m.BR_sa[t]))*mass*9.8
)
m.F_BL_y_ode = Constraint(m.t, rule=lambda m, t: 
	m.F_BL_y[t] == -D_f*sin(C_f*atan(B_f*m.BL_sa[t]))*mass*9.8
)

m.pc = ConstraintList()
m.pc.add(m.x[0]==0)
m.pc.add(m.y[0]==0)
m.pc.add(m.x_dot[0]==0)
m.pc.add(m.y_dot[0]==0)
m.pc.add(m.phi[0]==0)
m.pc.add(m.omega[0]==0)
m.pc.add(m.omega_dot[0]==0)
m.pc.add(m.FR_theta[0]==0)
m.pc.add(m.FL_theta[0]==0)
m.pc.add(m.BR_theta[0]==0)
m.pc.add(m.BL_theta[0]==0)

def cost_function(m, t):
	y_wp = sin(m.x[t]*2)*2 + m.x[t]

	return 3.0*(m.x[t] - 15)**2 + 1000.0*(m.y[t]-y_wp)**2 + (m.omega[t]-math.radians(0))**2 + \
		0.1*((m.F_FR_y[t]**2) + (m.F_FL_y[t]**2) + (m.F_BR_y[t]**2) + (m.F_BL_y[t]**2)) + \
		0.01*((m.FR_theta_dot[t]**2) + (m.FL_theta_dot[t]**2) + (m.BR_theta_dot[t]**2) + (m.BL_theta_dot[t]**2))

	"""
	thrtle = 3
	angl = 0
	if t>2:
		thrtle = 0
		angl = 70
	return (m.F_FR_x[t]-thrtle)**2 + (m.F_FL_x[t]-thrtle)**2 + (m.F_BR_x[t]-thrtle)**2 + (m.F_BL_x[t]-thrtle)**2 + \
		+ (m.FR_theta[t]-math.radians(angl))**2 + (m.FL_theta[t]-math.radians(angl))**2 + (m.BR_theta[t]-math.radians(angl))**2+ (m.BL_theta[t]-math.radians(angl))**2
	"""

m.integral = Integral(m.t, wrt=m.t, rule=cost_function)
m.obj = Objective(expr=m.integral)

TransformationFactory('dae.finite_difference').apply_to(m, wrt=m.t, nfe=traj_samples)
solver = SolverFactory('ipopt')
solver.options['tol'] = 1E-5
solver.options['max_iter'] = 2000
solver.options['print_level'] = 5
solver.solve(m, tee=True)

x_res = np.array([m.x[t]() for t in m.t]).tolist()
y_res = np.array([m.y[t]() for t in m.t]).tolist()
phi_res = np.array([m.phi[t]() for t in m.t]).tolist()

fr_x_res = np.array([m.F_FR_x[t]() for t in m.t]).tolist()
fr_y_res = np.array([m.F_FR_y[t]() for t in m.t]).tolist()

fr_theta_res = np.array([m.FR_theta[t]() for t in m.t]).tolist()
fl_theta_res = np.array([m.FL_theta[t]() for t in m.t]).tolist()
br_theta_res = np.array([m.BR_theta[t]() for t in m.t]).tolist()
bl_theta_res = np.array([m.BL_theta[t]() for t in m.t]).tolist()

#print(np.array([math.degrees(m.FR_sa[t]()) for t in m.t]).tolist())
#print(np.array([m.F_FR_x[t]() for t in m.t]).tolist())

fig, ax = plt.subplots(figsize=(10,10))

plt.xlim([-5, 20])
plt.ylim([-5, 20])

patch_len = 0.5
patch_wid = 0.1

fr_patch = ax.add_patch(Rectangle((0,0), patch_len, patch_wid))
fl_patch = ax.add_patch(Rectangle((0,0), patch_len, patch_wid))
br_patch = ax.add_patch(Rectangle((0,0), patch_len, patch_wid))
bl_patch = ax.add_patch(Rectangle((0,0), patch_len, patch_wid))

x = np.arange(0, 15, 0.1)
y = 2.0*np.sin(x*2)+x
ax.plot(x, y)

for i, t in enumerate(m.t):
	print(t, m.v_x[t](), m.v_y[t](), m.F_FR_y[t](), math.degrees(m.FR_sa[t]()))

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
	fl_theta = math.degrees(fl_theta_res[i])
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