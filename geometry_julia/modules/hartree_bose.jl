using Optim
using ForwardDiff


"""Quantum phase transition line for the Hartree-Bose model. Prints only positive χ coordinate."""
qpt(λ::Real)::Real = λ>1 ? 0 : sqrt((λ-1)/(λ-2))


"""Hartree-Bose energy functional.

n - Hamiltonian dimension,
(λ,χ) - parameter space coordinate.
rho - generally Hartree-Bose wavefunction complex coefficient: |φΗΒ>=1/sqrt(n!) 1/sqrt(1+abs(rho)^2) a^+ + rho/sqrt(1+abs(rho)^2) b^2 |0>.
"""
function hb_energy_functional(n::Integer, λ::Real, χ::Real, rho::Union{Real,Complex})::Real
    term1 = 1/2 * (rho' * rho - 1.) / (1. + abs(rho)^2)
    term2 = 1 / ((1. + abs(rho)^2)^2) * (λ / 4 * (rho' * rho' + rho^2 + 2 * rho' * rho) + 
            χ * (rho' * rho' * rho + rho' * rho * rho) + χ^2 * (rho' * rho)^2)

    result = n * (term1 - term2)
    return result
end


"""Function to minimize, simplification of hb_energy was done in Mathematica (at the infinite limit).
(λ,χ) - parameter space coordinate."""
function fun(λ::Real, χ::Real, rho::Real)::Real
    return -(2 + 4λ*rho^2 + 8χ*rho^3 + rho^4 * (-2 + 4χ^2)) / (4(1 + rho^2)^2)
end


"""Find rho minimizing the Hartree-Bose energy functional.
(λ,χ) - parameter space coordinate
Returns:
    rho - the coefficient minimizing HB energy functional, see `hb_energy_functional`"""
function rho_optimal(λ::Real, χ::Real)::Real
    if abs(χ) < qpt(λ)
        return 0
    else
        # SimulatedAnnealing(), Optim.Options(g_tol = 1e-5)
        result = optimize(rho -> fun(λ, χ, rho), -100.0, 100.0)
        return result.minimizer
    end
end


"""Exact rho minimizing the Hartree-Bose energy functional at the infinite limit.
(λ,χ) - parameter space coordinate
Returns:
    rho - the coefficient minimizing HB energy functional, see `hb_energy_functional`"""
function rho_optimal_exact(λ::Real, χ::Real)::Real
    if λ<1.
        return 0
    else
        rhophase1(l, c) = -(1 + l - 2*c^2)/(3*c) - (-36*c^2 - 4*(1 + l - 2*c^2)^2) / (3*(2^(2/3))*c*(-16 - 48*l - 48*l^2 - 16*l^3 - 336*c^2 + 192*l*c^2 + 96*l^2*c^2 + 240*c^4 - 192*l*c^4 + 128*c^6 + sqrt((-16 - 48*l -48*l^2-16*l^3-336*c^2+192*l*c^2+96*l^2*c^2+240*c^4-192*l*c^4+128*c^6)^2 + 4*(-36*c^2 - 4*(1 + l - 2*c^2)^2)^3))^(1/3)) + 1/(6*2^(1/3)*c)*(-16 - 48*l - 48*l^2 - 16*l^3 - 336*c^2 + 192*l*c^2 + 96*l^2*c^2 + 240*c^4 - 192*l*c^4 + 128*c^6 + sqrt((-16 - 48*l - 48*l^2 - 16*l^3 - 336*c^2 + 192*l*c^2 + 96*l^2*c^2 + 240*c^4 - 192*l*c^4 + 128*c^6)^2 + 4*(-36*c^2 - 4*(1 + l - 2*c^2)^2)^3))^(1/3)

        if χ>0
            return rhophase1(λ, χ)
        else
            # not working idk why, fuck this shit
            # return rhophase1(λ, -χ)
            return 0
        end
    end
end
    



"""Hartree-Bose wavefunction in a basis |j, m> as a function of rho, which is the result of minimization at every lambda, chi."""
function hb_t(n::Integer, rho::Real)
    result = [
        sqrt(binomial(big(n), big(Int(n/2 + m)))) * rho^(n/2 + m) / (1 + rho^2)^(n/2)
        for m in n/2:-1:-n/2
    ]

    return result
end


"""Hartree-Bose energy at dimension n and coordinate (λ,χ).

n - Hamiltonian dimension,
(λ,χ) - parameter space coordinate.
t - generally Hartree-Bose wavefunction complex coefficient: |φΗΒ>=1/sqrt(n!) 1/sqrt(1+abs(t)^2) a^+ + t/sqrt(1+abs(t)^2) b^2 |0>.
"""
function hb_energy(n::Integer, λ::Real, χ::Real)::Real
    rho = rho_optimal(λ, χ)
    return hb_energy_functional(n, λ, χ, rho)
end


"""Hartree-Bose wavefunction in a basis |j, m> as a function of lambda and chi."""
function hb(n::Integer, λ::Real, χ::Real)::Array{Real, 1}
    rho = rho_optimal(λ, χ)
    return hb_t(n, rho)
end

using Statistics

"""Reducing noise in hb(n, λ, χ) by averaging over a small area.
n - Hamiltonian dimension,
(λ,χ) - parameter space coordinate,
average_over - number of points to average over,
d - distance from λ and χ up to which points are generated.
Returns v0.
"""
function hb_precise(n::Integer, λ::Real, χ::Real; average_over::Integer=10, d::Real=0.001)::Array{Real, 1}
    dd = d/average_over
    num = floor(sqrt(average_over/2))
    
    return mean([hb(n, λ+k*dd, χ+m*dd) for k in -num:num, m in -num:num])
end



"""Metric tensor of Hartree-Bose model. Holds for any real bosonic Hamiltonian as N->∞.
(λ,χ) - parameter space coordinate.
Returns:
    [g11, g12, g22] - metric tensor components.
"""
function metric_hb(λ::Real, χ::Real)::Array{Real, 1}
    rho(x::Vector) = rho_optimal(x[1], x[2])
    
    # redivative over λ and χ
    rho_λ, rho_χ = ForwardDiff.gradient(rho, [λ, χ])

    g11 = rho_λ*rho_λ
    g12 = rho_λ*rho_χ
    g22 = rho_χ*rho_χ
    return [g11,g12,g22] / (1+rho([λ, χ])^2)^2
end