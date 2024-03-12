
σ = [
    [0 1; 1 0], 
    [0 -im; im 0],
    [1 0; 0 -1],
]



"""
Calculates the eigenvalues and eigenvectors of a hermitian matrix and sorts them by ascending eigenvalue.
"""
function eigensystem_sorted(matrix)
    eigen_values, eigen_vectors = eigen(matrix)
    eigen_values = real(eigen_values)
    eigen_vectors = eigen_vectors[:, sortperm(eigen_values)]
    eigen_values = eigen_values[sortperm(eigen_values)]
    return eigen_values, eigen_vectors
end

"""
Compute the partial derivative of the Hamiltonian with respect to the coordinate index `idx`.
"""
function H_partial(coord, H, idx)
    jacre = ForwardDiff.jacobian(x -> vec(real(H(x...))), coord)
    jacim = ForwardDiff.jacobian(x -> vec(imag(H(x...))), coord)
    jac = jacre + im*jacim
    return reshape(jac[:,idx], size(H(coord...)))
end
# only real Hamiltonian
# function H_partial(coord, H, idx)
#     jac = ForwardDiff.jacobian(x -> vec(H(x...)), coord)
#     return reshape(jac[:,idx], size(H(coord...)))
# end

"""
Calculate the Fubini-Study metric tensor on the ground state manifold of a Hamiltonian H at specific coordinate.
# Arguments:
- `coord:: Array{Float64}`: The coordinates of the point at which the metric tensor is calculated.
- `H:: Function R^2->Matrix(NxN)`: The Hamiltonian.
# Returns:
- `metric:: Array{Float64}`: The metric tensor.
"""
function g(coord, H)
    dim = length(coord)
    H_matrix = H(coord...)
    eigen_values, eigen_vectors = eigensystem_sorted(H_matrix)
    psi_0 = eigen_vectors[:, 1]
    
    # Hamiltonian dimension
    nn = size(H_matrix)[1]
    metric = zeros(dim, dim)
    for a in 1:dim
        for b in 1:dim
            H_a = H_partial(coord, H, a)
            H_b = H_partial(coord, H, b)
            for i in 2:nn
                psi_i = eigen_vectors[:, i]
                term = real(psi_0' * H_a * psi_i * (psi_i' * H_b * psi_0))/(eigen_values[1] - eigen_values[i])^2
                metric[a, b] += term
            end
        end
    end
    return metric
end


"""Helping function needed for the calculation of the simplified Ricci scalar `R(coord, H)`."""
function Abracket(coord, H)
    x, y = coord
    h = 1e-5  # small step size for numerical differentiation
    # Compute metric tensor at points needed for derivatives
    g_xph = g([x + h, y], H)
    g_xmh = g([x - h, y], H)
    g_yph = g([x, y + h], H)
    g_ypmh = g([x, y - h], H)
    gg = g(coord, H)
    det_g = det(gg)
    sq = sqrt(abs(det_g))
    # Compute derivatives using precomputed metric tensors
    ∂1g22 = (g_xph[2,2] - g_xmh[2,2]) / (2 * h)
    ∂2g11 = (g_yph[1,1] - g_ypmh[1,1]) / (2 * h)
    
    return gg[1,2]/(gg[1,1]*sq) * ∂2g11 - ∂1g22/sq
end

"""Helping function needed for the calculation of the simplified Ricci scalar `R(coord, H)`."""
function Bbracket(coord, H)
    x, y = coord
    h = 1e-5  # small step size for numerical differentiation
    # Compute metric tensor at points needed for derivatives
    g_xph = g([x + h, y], H)
    g_xmh = g([x - h, y], H)
    g_yph = g([x, y + h], H)
    g_ypmh = g([x, y - h], H)
    gg = g(coord, H)
    det_g = det(gg)
    sq = sqrt(abs(det_g))
    # Compute derivatives using precomputed metric tensors
    ∂1g11 = (g_xph[1,1] - g_xmh[1,1]) / (2 * h)
    ∂1g12 = (g_xph[1,2] - g_xmh[1,2]) / (2 * h)
    ∂2g11 = (g_yph[1,1] - g_ypmh[1,1]) / (2 * h)

    return 2*∂1g12/sq - ∂2g11/sq - gg[1,2]/(gg[1,1]*sq) * ∂1g11
end

"""
Ricci scalar calculated using simplified formula for Riemannian 2D manifold.
"""
function R(coord, H)
    x, y = coord
    h = 1e-5  # small step size for numerical differentiation
    ∂1A = (Abracket([x + h, y], H) - Abracket([x - h, y], H)) / (2 * h)
    ∂2B = (Bbracket([x, y + h], H) - Bbracket([x, y - h], H)) / (2 * h)
    
    (∂1A+∂2B)/sqrt(abs(det(g(coord, H))))
end




# print functions to std
all_names = names(Main, all=true, imported=true)
function_names = filter(name -> isa(getfield(Main, name), Function), all_names)
print(function_names)