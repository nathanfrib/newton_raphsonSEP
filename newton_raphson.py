import numpy as np

def newton_raphson(F, J, x0, tol=1e-6, max_iter=100):
    """
    Implementação do método de Newton-Raphson para resolver sistemas de equações não lineares.
    
    Parâmetros:
        F       : função que recebe o vetor x e retorna o vetor de funções (F(x) = 0)
        J       : função que recebe o vetor x e retorna a matriz Jacobiana associada a F
        x0      : vetor de chute inicial
        tol     : tolerância para critério de convergência
        max_iter: número máximo de iterações
        
    Retorna:
        x       : vetor solução aproximada
    """
    x = np.array(x0, dtype=float)
    
    for i in range(max_iter):
        Fx = np.array(F(x), dtype=float)

        # Verifica se a norma do vetor de funções está abaixo da tolerância
        if np.linalg.norm(Fx, ord=np.inf) < tol:
            print(f"Convergência alcançada após {i} iterações.")
            return x
        
        Jx = np.array(J(x), dtype=float)
        
        # Resolve o sistema linear: J(x) * dx = -F(x)
        try:
            dx = np.linalg.solve(Jx, -Fx)
        except np.linalg.LinAlgError as e:
            raise Exception("A matriz Jacobiana é singular ou mal condicionada.") from e
        
        x += dx
        print(f"iteração {i}: x = {x}; F(x) = {Fx}")        
    raise Exception(f"O método de Newton-Raphson não convergiu em {max_iter} iterações.")

# ================================
# Exemplo de aplicação:
# Sistema de equações para ilustração:
#   f1(x, y) = x² + y² - 4 = 0
#   f2(x, y) = x - y = 0
# OBS: Substitua estas funções pelas suas equações de fluxo de potência!
# ================================

def F_exemplo(x):
    f1 = x[0]**2 + x[1]**2 - 4
    f2 = x[0] - x[1]
    return [f1, f2]

def J_exemplo(x):
    # Derivadas parciais de f1 e f2
    df1_dx = 2 * x[0]
    df1_dy = 2 * x[1]
    df2_dx = 1
    df2_dy = -1
    return [[df1_dx, df1_dy],
            [df2_dx, df2_dy]]

if __name__ == "__main__":
    # Estimativa inicial
    x0 = [1.0, 1.0]
    
    # Executa o método de Newton-Raphson
    sol = newton_raphson(F_exemplo, J_exemplo, x0)
    
    print("Solução encontrada:", sol)
