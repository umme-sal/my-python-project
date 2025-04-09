import math
import cmath

class MatrixOperations:
    
    @staticmethod
    def scaler_multi(matrix,k):
        matrix_new=[]
        for i in range(len(matrix)):
            matrix_col=[]
            for j in range(len(matrix[0])):
                matrix_col.append(matrix[i][j]*k)
            matrix_new.append(matrix_col)
        return matrix_new

    @staticmethod
    def transpose(matrix):
        return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

    @staticmethod
    def get_minor(matrix, i, j):
        return [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]

    @staticmethod
    def determinant(matrix):
        if len(matrix) == 1:
            return matrix[0][0]
        
        if len(matrix) == 2:
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        
        det = 0
        for c in range(len(matrix)):
            det += ((-1) ** c) * matrix[0][c] * MatrixOperations.determinant(MatrixOperations.get_minor(matrix, 0, c))
        return det

    @staticmethod
    def matrix_multiply(A, B):
        rows_A, cols_A = len(A), len(A[0])
        rows_B, cols_B = len(B), len(B[0])

        if cols_A != rows_B:
            raise ValueError("Number of columns in A must be equal to number of rows in B")

        result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
        
        for i in range(rows_A):
            for j in range(cols_B):
                for k in range(cols_A):
                    result[i][j] += A[i][k] * B[k][j]

        return result

    @staticmethod
    def cofactor_matrix(matrix):
        cofactors = []
        for i in range(len(matrix)):
            cofactor_row = []
            for j in range(len(matrix)):
                minor = MatrixOperations.get_minor(matrix, i, j)
                cofactor_row.append(((-1) ** (i + j)) * MatrixOperations.determinant(minor))
            cofactors.append(cofactor_row)
        return cofactors

    @staticmethod
    def adjoint_matrix(matrix):
        cofactors = MatrixOperations.cofactor_matrix(matrix)
        return MatrixOperations.transpose(cofactors)

    @staticmethod
    def row_echelon(matrix):
        A = [row[:] for row in matrix]
        
        rows, cols = len(A), len(A[0])
        lead = 0
        
        for r in range(rows):
            if lead >= cols:
                return A
            i = r
            while A[i][lead] == 0:
                i += 1
                if i == rows:
                    i = r
                    lead += 1
                    if cols == lead:
                        return A
            A[i], A[r] = A[r], A[i]
            lv = A[r][lead] 
            A[r] = [mrx / float(lv) for mrx in A[r]]

            for i in range(rows):
                if i != r:
                    lv = A[i][lead]
                    A[i] = [round(iv - lv * rv,2) for rv, iv in zip(A[r], A[i])]
            
            lead += 1
            for i in range(rows):
                A[i] = [0 if abs(x) < 1e-10 else x for x in A[i]]
        return A

    @staticmethod
    def matrix_rank(matrix):
        ref_matrix = MatrixOperations.row_echelon(matrix)
        print(ref_matrix)
        return sum(1 for row in ref_matrix if any(row))

    @staticmethod
    def matrix_power(A, n):
        result = A
        for _ in range(1, n):
            result = MatrixOperations.matrix_multiply(result, A)
        return result

    @staticmethod
    def inverse_matrix(A):
        adj_A=MatrixOperations.adjoint_matrix(A)
        print("Adjoint:",adj_A)
        print("determinant:",MatrixOperations.determinant(A))
        if MatrixOperations.determinant(A)==0:
            return -1
        else:
            inv_A=MatrixOperations.scaler_multi(adj_A,1//MatrixOperations.determinant(A))
            return inv_A


    @staticmethod
    def lu_decomposition(A):
        n = len(A)
        L = [[0.0] * n for _ in range(n)]
        U = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for k in range(i, n):
                sum_ = sum(L[i][j] * U[j][k] for j in range(i))
                U[i][k] = A[i][k] - sum_

            for k in range(i, n):
                if i == k:
                    L[i][i] = 1
                else:
                    sum_ = sum(L[k][j] * U[j][i] for j in range(i))
                    L[k][i] = (A[k][i] - sum_) / U[i][i]

        return L, U
    
    @staticmethod
    def eigenvalues_2x2(matrix):
        a, b = matrix[0]
        c, d = matrix[1]

        trace = a + d
        determinant = MatrixOperations.determinant(matrix)
        
        # CE = n^2 - (trace)n + (determinant) = 0
        discriminant = trace**2 - 4 * determinant
        if discriminant < 0:
            return "Complex eigenvalues"
        
        lambda1 = (trace + math.sqrt(discriminant)) / 2
        lambda2 = (trace - math.sqrt(discriminant)) / 2
        
        return lambda1, lambda2
    
    @staticmethod
    def eigenvalues_3x3(matrix):
        a, b, c = matrix[0]
        d, e, f = matrix[1]
        g, h, i = matrix[2]
        
        trace = a + e + i
        determinant = MatrixOperations.determinant(matrix)
        p = a*e + a*i + e*i - (b*f + c*h + d*g)

        # CE: x^3 - trace*x^2 + p*x - determinant = 0
        coeffs = [1, -trace, p, -determinant]
        
        # Roots of the characteristic polynomial
        eigenvalues = MatrixOperations.solve_cubic(coeffs)
        return eigenvalues
    
    @staticmethod
    def solve_cubic(coeffs):
        #using Cardano's formula
        a, b, c, d = coeffs

        #  ax^3 + bx^2 + cx + d = 0 to x^3 + px + q=0
        p = c/a - (b/a)**2 / 3
        q = (2*(b/a)**3 / 27) - (b*c)/(3*a**2) + d/a

        #Discriminant
        discriminant = (q**2 / 4) + (p**3 / 27)


        # One real root and two complex roots
        if discriminant > 0:  
            u = (-q/2 + cmath.sqrt(discriminant))**(1/3)
            v = (-q/2 - cmath.sqrt(discriminant))**(1/3)
            root1 = u + v - (b/(3*a))
            root2 = -(u + v)/2 - (b/(3*a)) + cmath.sqrt(3)*(u - v)/2j
            root3 = -(u + v)/2 - (b/(3*a)) - cmath.sqrt(3)*(u - v)/2j
            return [root1, root2, root3]
        

        # All roots real, at least two are equal
        elif discriminant == 0:  
            u = (-q/2)**(1/3)
            root1 = 2*u - (b/(3*a))
            root2 = -u - (b/(3*a))
            root3 = root2
            return [root1, root2, root3]
        

        # All roots real and unequal
        else:  
            theta = math.acos(-q/2 / math.sqrt(-(p/3)**3))
            root1 = 2*math.sqrt(-p/3) * math.cos(theta/3) - (b/(3*a))
            root2 = 2*math.sqrt(-p/3) * math.cos((theta + 2*math.pi) / 3) - (b/(3*a))
            root3 = 2*math.sqrt(-p/3) * math.cos((theta + 4*math.pi) / 3) - (b/(3*a))
            return [root1, root2, root3]
        


A = [[2, 3, 4], [5, 6, 7], [8, 9, 10]]
B = [[5, 6, 6], [7, 8, 8], [1, 23, 3]]
C=[[4, 7, 2],
    [3, 6, 1],
    [2, 5, 1]]

D=[[1,2],[3,4]]

# B = [[5, 6, 6], [7, 8, 8], [1, 23, 3]]
# D=[[4,1],[2,3]]

# r=int(input("Enter the number of rows:"))
# c=int(input("Enter the number of columns:"))
# print("Enter the matrix:")
# A=[]
# for i in range(r):
#     rows=[]
#     for j in range(c):
#         n=eval(input())
#         rows.append(n)
#     A.append(rows)

# print(A)


print("Inverse of A:")
print(MatrixOperations.inverse_matrix(A))

print("Adjoint of A:")
print(MatrixOperations.adjoint_matrix(A))


print("Transpose of A:")
print(MatrixOperations.transpose(A))

print("Determinant of A:")
print(MatrixOperations.determinant([[6, 1, 1], [4, -2, 5], [2, 8, 7]]))

#print("Matrix Multiplication of A and B:")
#print(MatrixOperations.matrix_multiply(A, B))

print("Rank of A:")
print(MatrixOperations.matrix_rank(A))

print("LU Decomposition of A:")
L, U = MatrixOperations.lu_decomposition([[1, 1, 1, 1], [4, 3, -1, 1], [3, 5, 3, 1],[1 ,1 ,1 ,1]])
print("L:", L)
print("U:", U)

print("row echelon form of A:",MatrixOperations.row_echelon(A))


print("Eigenvalues of 2x2 matrix: ",MatrixOperations.eigenvalues_2x2(D))

print("Eigenvalues of 3x3 matrix: ",MatrixOperations.eigenvalues_3x3(A))

