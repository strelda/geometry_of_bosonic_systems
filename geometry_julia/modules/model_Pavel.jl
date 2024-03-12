using LinearAlgebra

""" General Lipkin model in the form H = a + jx * Jx + jy * Jy + ... """
function Lipkin(N; a=0, jx=0, jz=0, jxx=0, jzz=0, jxz=0)
    dim = N + 1
    result = Matrix{Float64}(I, dim, dim)

    # Jx = 0.5 * (J+ + J-)
    # Jxx = 0.25 * (J++ + J+- + J-+ + J--)
    # Jxz = 0.5 * (J+ Jz + Jz J+ + J- Jz + Jz J-)

    for i = 1:dim
        m = i - 1 - 0.5 * N
        result[i, i] = a - jz * m + jzz * m * m + 0.25 * jxx * (ShiftMinus(N, i + 1) * ShiftPlus(N, i) + ShiftPlus(N, i - 1) * ShiftMinus(N, i))

        if i > 1
            r = 0.5 * ShiftMinus(N, i) * (jx - jxz * (2 * m - 1))
            result[i - 1, i] = r
            result[i, i - 1] = r
        end

        if i < dim
            r = 0.5 * ShiftPlus(N, i) * (jx - jxz * (2 * m + 1))
            result[i + 1, i] = r
            result[i, i + 1] = r
        end

        if i > 2
            r = 0.25 * jxx * ShiftMinus(N, i - 1) * ShiftMinus(N, i)
            result[i - 2, i] = r
            result[i, i - 2] = r
        end

        if i < dim - 1
            r = 0.25 * jxx * ShiftPlus(N, i + 1) * ShiftPlus(N, i)
            result[i + 2, i] = r
            result[i, i + 2] = r
        end
    end

    return Symmetric(result)
end

""" Raising operator """
function ShiftPlus(N, i)
    if i < 0 || i > N
        return 0
    end

    return sqrt((N - i + 1) * i)
end

""" Lowering operator """
function ShiftMinus(N, i)
    if i <= 0 || i > N + 1
        return 0
    end

    return sqrt((i - 1) * (N - i + 2))
end

""" Lipkin model from pavel """
function Model(λ, χ, size)
    N, = size

    jx = -χ
    jz = 1 - χ * χ
    jxx = -λ / N
    jxz = -χ / N
    jzz = -χ * χ / N
    a = 0.5 * N * (1 - χ * χ)

    return Lipkin(N, a=a, jx=jx, jz=jz, jxx=jxx, jxz=jxz, jzz=jzz)
end

function ModelDerivative(λ, χ, size)
    N, = size

    v1 = Lipkin(N, jxx=-1/N)
    v2 = Lipkin(N, jxz=-1/N, jx=-1)
    v3 = Lipkin(N, a=-N/2, jz=-1, jzz=-1/N)

    h1 = v1
    h2 = v2 + 2 * χ * v3

    return h1, h2
end

function ModelName()
    return "Matus"
end


