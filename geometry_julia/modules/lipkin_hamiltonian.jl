using SparseArrays
using Arpack


# Hamiltonian

Je(sn, j, m) = sqrt(j * (j + 1) - m * (m + sn * 1))

"""Lipkin model Hamiltonian in dimension n at point λ, χ.
n - Hamiltonian dimension,
(λ,χ) - parameter space coordinate.
"""
function HConstructor(n::Integer, λ::Real, χ::Real)
    diagonal_entries = [m - (1 / n) * (
                    λ/4 *(Je(-1, n/2, m) * Je(1, n/2, m-1) + Je(1, n/2, m) * Je(-1, n/2, m+1))  +  χ^2 * (n/2 + m)^2 ) 
                    for m in n/2:-1:-n/2]
    offdiagonal_entries = [- (χ / 2n) * ((m + n/2) * Je(1, n/2, m) + (m + 1 + n/2) * Je(1, n/2, m)) 
                    for m in n/2-1:-1:-n/2]
    offdiagonal2_entries = [-(1/n) * λ * (1/4) * Je(1, n/2, m+1) * Je(1, n/2, m) 
                    for m in n/2-2:-1:-n/2]
    
    diagonal = spdiagm(0 => diagonal_entries)
    offdiagonal = spdiagm(1 => offdiagonal_entries, -1 => offdiagonal_entries)
    offdiagonal2 = spdiagm(2 => offdiagonal2_entries, -2 => offdiagonal2_entries)

    return diagonal + offdiagonal + offdiagonal2
end


"""Derivative dH/dλ of the Lipkin model Hamiltonian in dimension n at point λ, χ.
n - Hamiltonian dimension,
(λ,χ) - parameter space coordinate.
"""
function DH1Constructor(n::Integer, λ::Real, χ::Real)
    diagonal_entries = [- (1 / n) * (
                    1/4 *(Je(-1, n/2, m) * Je(1, n/2, m-1) + Je(1, n/2, m) * Je(-1, n/2, m+1))) 
                    for m in n/2:-1:-n/2]
    offdiagonal2_entries = [-(1/n) * (1/4) * Je(1, n/2, m+1) * Je(1, n/2, m) 
                    for m in n/2-2:-1:-n/2]

    diagonal = spdiagm(0 => diagonal_entries)
    offdiagonal2 = spdiagm(2 => offdiagonal2_entries, -2 => offdiagonal2_entries)

    return diagonal + offdiagonal2
end

"""Derivative dH/dχ of the Lipkin model Hamiltonian in dimension n at point λ, χ.
n - Hamiltonian dimension,
(λ,χ) - parameter space coordinate.
"""
function DH2Constructor(n::Integer, λ::Real, χ::Real)
    diagonal_entries = [- (1 / n) * (2 * χ * (n/2 + m)^2 ) 
                    for m in n/2:-1:-n/2]
    offdiagonal_entries = [- (1 / 2n) * ((m + n/2) * Je(1, n/2, m) + (m + 1 + n/2) * Je(1, n/2, m)) 
                    for m in n/2-1:-1:-n/2]

    diagonal = spdiagm(0 => diagonal_entries)
    offdiagonal = spdiagm(1 => offdiagonal_entries, -1 => offdiagonal_entries)

    return diagonal + offdiagonal
end




# ground state

"""Ground state energy e0 and vector v0 of Lipkin model Hamiltonian (see HConstructor) in dimension n at point λ, χ.
n - Hamiltonian dimension,
(λ,χ) - parameter space coordinate.
Returns (e0, v0).
"""
function gs(n::Integer, λ::Real, χ::Real)::Tuple{Real, Array{Real, 1}}
    H = HConstructor(n, λ, χ)
    e, v = eigs(H, nev=1, which=:SR)
    
    if v[:,1][1]>0
        return e[1], v[:,1]
    else
        return e[1], -v[:,1]
    end
end


using Statistics

"""Reducing noise in gs(n, λ, χ) by averaging over a small area.
n - Hamiltonian dimension,
(λ,χ) - parameter space coordinate,
average_over - number of points to average over,
d - distance from λ and χ up to which points are generated.
Returns v0.
"""
function gs_precise(n::Integer, λ::Real, χ::Real; average_over::Integer=10, d::Real=0.001)::Tuple{Real, Array{Real, 1}}
    dd = d/average_over
    num = floor(sqrt(average_over/2))
    e = mean([gs(n, λ+k*dd, χ+m*dd)[1] for k in -num:num, m in -num:num])
    v = mean([gs(n, λ+k*dd, χ+m*dd)[2] for k in -num:num, m in -num:num])
    return (e,v)
end


"""Intensive metric tensor of ground state manifold for Lipkin model (see HConstructor) at coordinate (λ,χ).
(λ,χ) - parameter space coordinate.
Returns:
    [g11, g12, g22] - metric tensor components.
"""
function metric_lipkin_intensive(n::Integer, λ::Real, χ::Real; average_over::Integer=10, d::Real=0.001)::Array{Real, 1}
    d = 1e-2
    # vector of ground state wave function
    # gsn(x::Vector)::Array{Real, 1} = gs(n, x[1], x[2])[2]
    gsn(x::Vector)::Array{Real, 1} = gs_precise(n, x[1], x[2],average_over=average_over, d=d)[2]

    # # 1. order derivative in λ and χ direction
    gs_λ = (gsn([λ+d, χ]) - gsn([λ-d, χ]))/(2*d)
    gs_χ = (gsn([λ, χ+d]) - gsn([λ, χ-d]))/(2*d)

    # richardson extrapolation
    # gs_λ = (gsn([λ+2*d, χ]) - 8*gsn([λ+d, χ]) + 8*gsn([λ-d, χ]) - gsn([λ-2*d, χ]))/(12*d)
    # gs_χ = (gsn([λ, χ+2*d]) - 8*gsn([λ, χ+d]) + 8*gsn([λ, χ-d]) - gsn([λ, χ-2*d]))/(12*d)

    g11 = dot(gs_λ,gs_λ)
    g12 = dot(gs_λ,gs_χ)
    g22 = dot(gs_χ,gs_χ)

    return [g11,g12,g22]/n
end


# geometry

"""
Calculate the Fubini-Study metric tensor on the ground state manifold of a Hamiltonian H in dimension n at point λ, χ.
n - Hamiltonian dimension,
(λ,χ) - parameter space coordinate.
# Returns:
- `metric:: Array{Float64}`: The metric tensor.
"""
function g(n::Integer, λ::Real, χ::Real)
    dim = 2 # number of Hamiltonian parameters
    H = HConstructor(n, λ, χ)
    DH = [DH1Constructor(n, λ, χ), DH2Constructor(n, λ, χ)]
    eigen_values, eigen_vectors = eigen(Matrix(H))
    psi_0 = eigen_vectors[:, 1]

    metric = zeros(dim, dim)
    for a in 1:dim
        for b in 1:dim
            for i in 2:n
                psi_i = eigen_vectors[:, i]
                term = real(psi_0' * DH[a] * psi_i * (psi_i' * DH[b] * psi_0))/(eigen_values[1] - eigen_values[i])^2
                metric[a, b] += term
            end
        end
    end
    return metric
end


"""Function needed for the calculation of the simplified Ricci scalar `R(coord, H)`."""
function Abracket(n::Integer, λ::Real, χ::Real)
    h = 1e-5  # small step size for numerical differentiation
    # Compute metric tensor at points needed for derivatives
    g_xph = g(n, λ+h, χ)
    g_xmh = g(n, λ-h, χ)
    g_yph = g(n, λ, χ+h)
    g_ymh= g(n, λ, χ-h)
    gg = g(n, λ, χ)
    det_g = det(gg)
    sq = sqrt(abs(det_g))
    # Compute derivatives using precomputed metric tensors
    ∂1g22 = (g_xph[2,2] - g_xmh[2,2]) / (2 * h)
    ∂2g11 = (g_yph[1,1] - g_ymh[1,1]) / (2 * h)
    
    return gg[1,2]/(gg[1,1]*sq) * ∂2g11 - ∂1g22/sq
end

"""Function needed for the calculation of the simplified Ricci scalar `R(coord, H)`."""
function Bbracket(n::Integer, λ::Real, χ::Real)
    h = 1e-5  # small step size for numerical differentiation
    # Compute metric tensor at points needed for derivatives
    g_xph = g(n, λ+h, χ)
    g_xmh = g(n, λ-h, χ)
    g_yph = g(n, λ, χ+h)
    g_ymh= g(n, λ, χ-h)
    gg = g(n, λ, χ)
    det_g = det(gg)
    sq = sqrt(abs(det_g))
    # Compute derivatives using precomputed metric tensors
    ∂1g11 = (g_xph[1,1] - g_xmh[1,1]) / (2 * h)
    ∂1g12 = (g_xph[1,2] - g_xmh[1,2]) / (2 * h)
    ∂2g11 = (g_yph[1,1] - g_ymh[1,1]) / (2 * h)

    return 2*∂1g12/sq - ∂2g11/sq - gg[1,2]/(gg[1,1]*sq) * ∂1g11
end

"""
Ricci scalar for Lipkin model calculated using simplified formula for Riemannian 2D manifold in dimension n at point λ, χ.
n - Hamiltonian dimension,
(λ,χ) - parameter space coordinate.
# Returns:
- `metric:: Array{Float64}`: The metric tensor.
"""
function R(n::Integer, λ::Real, χ::Real)
    h = 1e-5  # small step size for numerical differentiation
    ∂1A = (Abracket(n, λ+h, χ) - Abracket(n, λ-h, χ)) / (2 * h)
    ∂2B = (Bbracket(n, λ, χ+h) - Bbracket(n, λ, χ-h)) / (2 * h)
    
    (∂1A+∂2B)/sqrt(abs(det(g(n, λ, χ))))
end
