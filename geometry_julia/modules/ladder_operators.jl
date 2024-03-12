function krondelta(a, b)
    return(bool(a == b))
end

function Jp(x, a, b)
    if Int(2 * x + 1) == b
        return 0
    else
        tem = sqrt(x * (x + 1) - b * (b + 1))
        return tem * krondelta(a, b + 1)
    end
end

function Jm(x, a, b)
    tem = sqrt(x * (x + 1) - b * (b - 1))
    return tem * krondelta(a, b - 1)
end

function jz(n)
    A = zeros(Int, 2n + 1, 2n + 1)
    for i in 1:2n + 1
        A[i, i] = n - i + 1
    end
    return A
end

function jx(n)
    j = 2n + 1
    A = zeros(j, j)
    for i in 1:j
        for k in 1:j
            A[i, k] = 0.5 * (Jp(n, -n + i, -n + k) + Jm(n, -n + i, -n + k))
        end
    end
    return A
end

function jy(n)
    j = 2n + 1
    A = zeros(Complex{Float64}, j, j)
    for i in 1:j
        for k in 1:j
            A[i, k] = -1im * 0.5 * (Jm(n, -n + i, -n + k) - Jp(n, -n + i, -n + k))
        end
    end
    return A
end
