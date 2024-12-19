eval(@finch_kernel mode=:fast function mttkrp_dim3_finch_opt_helper_base(A_nondiag, B_T, C_T)
    C_T .= 0
    for l = _, k = _, i = _, j = _
        if and((i < k), (k < l))
            let B_T_jk = B_T[j, k], B_T_ji = B_T[j, i], B_T_jl = B_T[j, l], A_ikl = A_nondiag[i, k, l]
                C_T[j, l] += *(2, *(B_T_ji, A_ikl, B_T_jk))
                C_T[j, k] += *(2, *(B_T_ji, A_ikl, B_T_jl))
                C_T[j, i] += *(2, *(A_ikl, B_T_jl, B_T_jk))
            end
        end
    end
    return C_T
end)

eval(@finch_kernel mode=:fast function mttkrp_dim3_finch_opt_helper_edge(A_diag, B_T, C_T)
    C_T .= 0
    for l = _, k = _, i = _, j = _
        if and((identity(i) <= identity(k)), (identity(k) <= identity(l)))
            let kl_eq = (k == l), ik_eq = (i == k), B_T_jk = B_T[j, k], B_T_ji = B_T[j, i], B_T_jl = B_T[j, l], A_ikl = A_diag[i, k, l]
                if or(and(ik_eq, !kl_eq), and(!ik_eq, kl_eq))
                    C_T[j, l] += *(B_T_ji, A_ikl, B_T_jk)
                    C_T[j, k] += *(B_T_ji, A_ikl, B_T_jl)
                    C_T[j, i] += *(A_ikl, B_T_jl, B_T_jk)
                end
                if and(ik_eq, kl_eq)
                    C_T[j, i] += *(A_ikl, B_T_jl, B_T_jk)
                end
            end
        end
    end
    return C_T
end)

