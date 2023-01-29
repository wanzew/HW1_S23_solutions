using LinearAlgebra, Plots
import ForwardDiff as FD
import MeshCat as mc

function RotY(θ::Real)
    # rotation matrix for rotation about y axis
    return [cos(θ) 0 sin(θ); 0 1 0; -sin(θ) 0 cos(θ)]
end
function double_pendulum_dynamics(params::NamedTuple, x::Vector)
    # continuous time dynamics for a double pendulum

    # the state is the following:
    θ1,θ̇1,θ2,θ̇2 = x

    # system parameters
    m1, m2, L1, L2, g = params.m1, params.m2, params.L1, params.L2, params.g

    # dynamics
    c = cos(θ1-θ2)
    s = sin(θ1-θ2)

    ẋ = [
        θ̇1;
        ( m2*g*sin(θ2)*c - m2*s*(L1*c*θ̇1^2 + L2*θ̇2^2) - (m1+m2)*g*sin(θ1) ) / ( L1 *(m1+m2*s^2) );
        θ̇2;
        ((m1+m2)*(L1*θ̇1^2*s - g*sin(θ2) + g*sin(θ1)*c) + m2*L2*θ̇2^2*s*c) / (L2 * (m1 + m2*s^2));
        ]

    return ẋ
end
function backward_euler(params::NamedTuple, dynamics::Function, x1::Vector, x2::Vector, dt::Real)::Vector
    x1 + dt*dynamics(params,x2) - x2
end
function implicit_integrator_solve(params::NamedTuple, dynamics::Function, implicit_integrator::Function, x1::Vector, dt::Real)::Vector

    # initialize guess
    x2 = 1*x1

    # newton's method to solve for x2 such that residual(x2) = 0
    residual_fx(_x2) = implicit_integrator(params, dynamics, x1, _x2, dt)
    for i = 1:10
        residual = residual_fx(x2)
        if norm(residual)<1e-10
            return x2
        end
        residual_jacobian = FD.jacobian(residual_fx, x2)
        Δx2 = -residual_jacobian\residual
        x2 = x2 + Δx2
    end
    error("implicit integrator solve failed")
end

function forward_euler(params::NamedTuple, dynamics::Function, x::Vector, dt::Real)::Vector
    return x + dt*dynamics(params,x)
end
function midpoint(params::NamedTuple, dynamics::Function, x::Vector, dt::Real)::Vector
    xm = x + 0.5*dt*dynamics(params,x)
    return x + dt*dynamics(params,xm)
end
function rk4(params::NamedTuple, dynamics::Function, x::Vector, dt::Real)::Vector
    k1 = dt * dynamics(params, x)
    k2 = dt * dynamics(params, x + k1 / 2)
    k3 = dt * dynamics(params, x + k2 / 2)
    k4 = dt * dynamics(params, x + k3)
    return x + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
end

function double_pendulum_energy(params::NamedTuple, x::Vector)::Real
    θ1, θ̇1, θ2, θ̇2 = x
    m1, m2, L1, L2, g = params.m1, params.m2, params.L1, params.L2, params.g

    r1 = [L1*sin(θ1), 0, -params.L1*cos(θ1) + 2]
    r2 = r1 + [params.L2*sin(θ2), 0, -params.L2*cos(θ2)]
    v1 = [L1*θ̇1*cos(θ1), 0, L1*θ̇1*sin(θ1)]
    v2 = v1 + [L2*θ̇2*cos(θ2), 0, L2*θ̇2*sin(θ2)]

    kinetic = 0.5*(m1*v1'*v1 + m2*v2'*v2)
    potential = m1*g*r1[3] + m2*g*r2[3]
    return kinetic + potential
end
function simulate_explicit(params::NamedTuple,dynamics::Function,integrator::Function,x0::Vector,dt::Real,tf::Real)
    t_vec = 0:dt:tf
    N = length(t_vec)
    X = [zeros(length(x0)) for i = 1:N]
    X[1] = x0
    for i = 1:(N-1)
        X[i+1] = integrator(params,dynamics,X[i],dt)
    end
    E = [double_pendulum_energy(params,x) for x in X]
    return X, E
end
function simulate_implicit(params::NamedTuple,dynamics::Function,implicit_integrator::Function,x0::Vector,dt::Real,tf::Real)
    t_vec = 0:dt:tf
    N = length(t_vec)
    X = [zeros(length(x0)) for i = 1:N]
    X[1] = x0
    for i = 1:(N-1)
        X[i+1] = implicit_integrator_solve(params, dynamics, implicit_integrator, X[i], dt)
    end
    E = [double_pendulum_energy(params,x) for x in X]
    return X, E
end
function compare_explicit_integrator_speeds(params::NamedTuple,dynamics::Function,integrator::Function,x0::Vector,tf::Real)
    t_vec_10hz = 0:0.1:tf
    t_vec_100hz = 0:0.01:tf
    t_vec_1000hz = 0:0.001:tf
    X_10hz,   E_10hz   = simulate_explicit(params,dynamics,integrator,x0,0.1,  tf)
    X_100hz,  E_100hz  = simulate_explicit(params,dynamics,integrator,x0,0.01, tf)
    X_1000hz, E_1000hz = simulate_explicit(params,dynamics,integrator,x0,0.001,tf)
    plot([t_vec_10hz, t_vec_100hz, t_vec_1000hz],
         [E_10hz, E_100hz, E_1000hz],
         title = String(Symbol(integrator)) *  "(explicit)" , xlabel = "Time (s)", ylabel = "Energy",
         label = ["10hz" "100hz" "1000hz"])
end
function compare_implicit_integrator_speeds(params::NamedTuple,dynamics::Function,integrator::Function,x0::Vector,tf::Real)
    t_vec_10hz = 0:0.1:tf
    t_vec_100hz = 0:0.01:tf
    t_vec_1000hz = 0:0.001:tf
    X_10hz,   E_10hz   = simulate_implicit(params,dynamics,integrator,x0,0.1,  tf)
    X_100hz,  E_100hz  = simulate_implicit(params,dynamics,integrator,x0,0.01, tf)
    X_1000hz, E_1000hz = simulate_implicit(params,dynamics,integrator,x0,0.001,tf)
    plot([t_vec_10hz, t_vec_100hz, t_vec_1000hz],
         [E_10hz, E_100hz, E_1000hz],
         title =  String(Symbol(integrator)) * " (implicit)", xlabel = "Time (s)", ylabel = "Energy",
         label = ["10hz" "100hz" "1000hz"])
end
# vis = mc.Visualizer()
# mc.open(vis)
let

    # initial condition
    x0 = [pi/1.6; 0; pi/1.8; 0]

    params = (
        m1 = 1.0,
        m2 = 1.0,
        L1 = 1.0,
        L2 = 1.0,
        g = 9.8
    )

    # dt = 0.01
    tf = 1
    # compare_explicit_integrator_speeds(params,double_pendulum_dynamics,forward_euler,x0,tf)
    # compare_explicit_integrator_speeds(params,double_pendulum_dynamics,midpoint,x0,tf)
    # compare_explicit_integrator_speeds(params,double_pendulum_dynamics,rk4,x0,tf)
    compare_implicit_integrator_speeds(params,double_pendulum_dynamics,backward_euler,x0,tf)

    # X,E = simulate_implicit(params,double_pendulum_dynamics,backward_euler,x0,0.01,0.05)
    # t_vec_10hz = 0:0.1:tf
    # t_vec_100hz = 0:0.01:tf
    # t_vec_1000hz = 0:0.001:tf
    # plot(E)



    # X_euler_10hz,   E_euler_10hz   = simulate(params,double_pendulum_dynamics,forward_euler,x0,0.1,  tf)
    # X_euler_100hz,  E_euler_100hz  = simulate(params,double_pendulum_dynamics,forward_euler,x0,0.01, tf)
    # X_euler_1000hz, E_euler_1000hz = simulate(params,double_pendulum_dynamics,forward_euler,x0,0.001,tf)
    # plot([t_vec_10hz, t_vec_100hz, t_vec_1000hz],
    #      [E_euler_10hz, E_euler_100hz, E_euler_1000hz],
    #      title = "Forward Euler", xlabel = "Time (s)", ylabel = "Energy",
    #      label = ["10hz" "100hz" "1000hz"])
    # X_euler    = simulate(params,double_pendulum_dynamics,forward_euler,x0,dt,tf)
    # X_midpoint = simulate(params,double_pendulum_dynamics,midpoint,x0,dt,tf)
    # X_rk4      = simulate(params,double_pendulum_dynamics,rk4,x0,dt,tf)

    # E_euler    = [double_pendulum_energy(params,x) for x in X_euler]
    # E_midpoint = [double_pendulum_energy(params,x) for x in X_midpoint]
    # E_rk4      = [double_pendulum_energy(params,x) for x in X_rk4]


    # rod1 = mc.Cylinder(mc.Point(0,0,-params.L1/2), mc.Point(0,0,params.L1/2), 0.05)
    # rod2 = mc.Cylinder(mc.Point(0,0,-params.L2/2), mc.Point(0,0,params.L2/2), 0.05)
    # mc.setobject!(vis[:rod1], rod1)
    # mc.setobject!(vis[:rod2], rod2)
    #
    # # display(plot(t_vec, hcat(X...)'))
    # sphere = mc.HyperSphere(mc.Point(0,0,0.0), 0.1)
    # mc.setobject!(vis[:s1], sphere)
    # mc.setobject!(vis[:s2], sphere)
    # anim = mc.Animation(floor(Int,1/dt))
    # for k = 1:N
    #     mc.atframe(anim, k) do
    #         # mc.settransform!(vis[:box1], mc.Translation([X[k][1];0;0]))
    #         # mc.settransform!(vis[:box2], mc.Translation([X[k][3];0;0]))
    #         θ1,θ2 = X[k][[1,3]]
    #         r1 = [params.L1*sin(θ1), 0, -params.L1*cos(θ1) + 2]
    #         r2 = r1 + [params.L2*sin(θ2), 0, -params.L2*cos(θ2)]
    #         mc.settransform!(vis[:s1], mc.Translation(r1))
    #         mc.settransform!(vis[:s2], mc.Translation(r2))
    #         mc.settransform!(vis[:rod1], mc.compose(mc.Translation(0.5*([0,0,2] + r1)),mc.LinearMap(RotY(-θ1))))
    #         mc.settransform!(vis[:rod2], mc.compose(mc.Translation(r1 + 0.5*(r2 - r1)),mc.LinearMap(RotY(-θ2))))
    #     end
    # end
    # mc.setanimation!(vis, anim)
#
    # E = [double_pendulum_energy(params,x) for x in X]
#     @show E[1]
#     @show E[end]
#

    # @show abs(E[1] - E[end])
    # E0 = E_rk4[1] + .1
    # E_err_euler = [abs(e - E0)/E0 for e in E_euler]
    # E_err_midpoint = [abs(e - E0)/E0 for e in E_midpoint]
    # E_err_rk4 = [abs(e - E0)/E0 for e in E_rk4]
    # plot(t_vec, [E_euler,E_midpoint,E_rk4])
    # plot(t_vec, [E_euler,E_midpoint,E_rk4], xlabel = "time (s)",ylabel = "Energy",label = ["forward Euler" "midpoint" "rk4"])
    # plot(t_vec, [E_err_euler,E_err_midpoint,E_err_rk4], xlabel = "time (s)",ylabel = "Energy",label = ["forward Euler" "midpoint" "rk4"],yaxis=:log)

end
