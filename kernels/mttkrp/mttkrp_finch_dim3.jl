using Finch
import Finch.FinchNotation: and, or
using BenchmarkTools

A = Tensor(Dense(SparseList(SparseList(Element(0.0)))))
A_nondiag = Tensor(Dense(SparseList(SparseList(Element(0.0)))))
A_diag = Tensor(Dense(SparseList(SparseList(Element(0.0)))))
B = Tensor(Dense(Dense(Element(0.0))))   
B_T = Tensor(Dense(Dense(Element(0.0)))) 
C = Tensor(Dense(Dense(Element(0.0))))
C_T = Tensor(Dense(Dense(Element(0.0))))

include("../../SySTeC/generated/mttkrp_dim3.jl")

eval(@finch_kernel mode=:fast function mttkrp_finch_ref_dim3_helper(C_T, A, B_T)
    C_T .= 0
    for l=_, k=_, i=_, j=_
        C_T[j, i] += A[i, k, l] * B_T[j, l] * B_T[j, k]
    end
    return C_T
end)

function mttkrp_finch_ref_dim3(C, A, B)
    (n, r) = size(C)
    _C_T = Tensor(Dense(Dense(Element(0.0))), zeros(r, n))
    _A = Tensor(Dense(SparseList(SparseList(Element(0.0)))), A)  
    _B_T = Tensor(Dense(Dense(Element(0.0))), B) 
    @finch mode=:fast begin 
        _B_T .= 0
        for j=_, i=_ 
            _B_T[j, i] = B[i, j] 
        end 
    end  

    time = @belapsed mttkrp_finch_ref_dim3_helper($_C_T, $_A, $_B_T)
    _C = Tensor(Dense(Dense(Element(0.0))), C)
    @finch begin 
        _C .= 0
        for j=_, i=_
            _C[i, j] = _C_T[j, i]
        end
    end
    
    return (;time = time, C = _C)
end

function mttkrp_finch_opt_dim3(C, A, B)
    (n, n, n) = size(A)
    (n, r) = size(C)
    _C_T_nondiag = Tensor(Dense(Dense(Element(0.0))), zeros(r, n))
    _C_T_diag = Tensor(Dense(Dense(Element(0.0))), zeros(r, n))

    nondiagA = zeros(n, n, n)
    diagA = zeros(n, n, n)
    for k=1:n, j=1:n, i=1:n
        if i != j && j != k && i != k
            nondiagA[i, j, k] = A[i, j, k]
        end
        if i == j || j == k || i == k
            diagA[i, j, k] = A[i, j, k]
        end
    end
    _A_nondiag = Tensor(Dense(SparseList(SparseList(Element(0.0)))), nondiagA)
    _A_diag = Tensor(Dense(SparseList(SparseList(Element(0.0)))), diagA)

    _B_T = Tensor(Dense(Dense(Element(0.0))), B) 
    @finch mode=:fast begin 
        _B_T .= 0
        for j=_, i=_ 
            _B_T[j, i] = B[i, j] 
        end 
    end

    time_1 = @belapsed mttkrp_dim3_finch_opt_helper_base($_A_nondiag, $_B_T, $_C_T_nondiag)
    time_2 = @belapsed mttkrp_dim3_finch_opt_helper_edge($_A_diag, $_B_T, $_C_T_diag)
    C_full = Tensor(Dense(Dense(Element(0.0))), C)
    @finch mode=:fast for i=_, j=_
        C_full[i, j] = _C_T_nondiag[j, i] + _C_T_diag[j, i]
    end
    return (;time = time_1 + time_2, C = C_full, nondiag_time = time_1, diag_time = time_2)
end