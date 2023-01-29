using LinearAlgebra, Plots
import ForwardDiff as FD


function dynamics(params, x, u)
    # unpack state
    position = x[1]
    velocity = x[2]

    # model params
    k, b, mass = params.k, params.b, params.mass

    # accel
    acceleration = (u - k * position - b * velocity) / mass
    return [velocity; acceleration]
end
# function dynamics(params, )
# function simulate_T(params, dynamics, integrator, x0, T, dt)
#     @assert abs((T/dt) - round(T/dt)) == 0
#     N = T/dt # number of steps
#     X = [zeros(length(x0)) for i = 1:N]
#     X[1] = x0
#     for i = 1:(N-1)
#         X[i+1] = integrator(params, dynamics,
function rk4(params, dynamics, x, u, dt)
    k1 = dt*dynamics(params, x,u)
    k2 = dt*dynamics(params, x + k1/2, u)
    k3 = dt*dynamics(params, x + k2/2, u)
    k4 = dt*dynamics(params, x + k3, u)
    return x + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
end
function forward_euler(params, dynamics, x, u, dt)
    return x + dt*dynamics(params,x,u)
end
function energy(params, x)
    position = x[1]
    velocity = x[2]

    potential = 0.5*params.k*position^2
    kinetic = 0.5*params.mass*velocity^2

    return potential + kinetic
end
let

    dt = 0.01
    tf = 10
    N = Int(tf/dt)

    X = [zeros(2) for i = 1:N]
    X[1] = [3,0]
    params = (
        k = 5.0,
        b = 0.2,
        mass = 2.0
    )
    for i = 1:(N-1)
        X[i+1] = rk4(params, dynamics, X[i], 0, dt)
        # X[i+1] = forward_euler(params, dynamics, X[i], 0, dt)
    end

    plot(hcat(X...)')

    E = [energy(params,x) for x in X]

    plot(E)
end
