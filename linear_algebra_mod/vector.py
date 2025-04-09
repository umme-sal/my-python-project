import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

class VectorOperations:
    @staticmethod
    def cross_product(a, b):
        return [
            (a[1] * b[2]) - (a[2] * b[1]),
            (a[2] * b[0]) - (a[0] * b[2]),
            (a[0] * b[1]) - (a[1] * b[0])
        ]
    
    @staticmethod
    def magnitude(vector):
        return round(math.sqrt(sum([math.pow(component, 2) for component in vector])), 4)

    # Norm of a vector
    @staticmethod
    def norm(vector, p=2):
        total = sum(abs(component) ** p for component in vector)
        return round(total ** (1 / p), 4)

    @staticmethod
    def unit_vector(vector):
        mag = VectorOperations.magnitude(vector)
        return [round(component / mag, 2) for component in vector]

    @staticmethod
    def vector_subtraction(a, b):
        return [a[i] - b[i] for i in range(len(a))]

    @staticmethod
    def vector_addition(a, b):
        return [a[i] + b[i] for i in range(len(a))]

    @staticmethod
    def scalar_product(vector, scalar):
        return [component * scalar for component in vector]

    @staticmethod
    def dot_product(a, b):
        return sum(a[i] * b[i] for i in range(len(a)))

    @staticmethod
    def hadamard_product(a, b):
        if len(a) != len(b):
            raise ValueError("Vectors must have the same length")
        return [a[i] * b[i] for i in range(len(a))]

    @staticmethod
    def outer_product(a, b):
        return [[a[i] * b[j] for j in range(len(b))] for i in range(len(a))]

    @staticmethod
    def scalar_triple_product(a, b, c):
        return VectorOperations.dot_product(a, VectorOperations.cross_product(b, c))

    @staticmethod
    def vector_triple_product(a, b, c):
        cross_bc = VectorOperations.cross_product(b, c)
        return VectorOperations.cross_product(a, cross_bc)

    # Linear combination of vectors (with scalars)
    @staticmethod
    def linear_combination(vectors, scalars):
        result = [0] * len(vectors[0])
        for i, vector in enumerate(vectors):
            result = VectorOperations.vector_addition(result, VectorOperations.scalar_product(vector, scalars[i]))
        return result

    # Angle between two vectors (in degrees)
    @staticmethod
    def angle_between_vectors(a, b):
        dot_ab = VectorOperations.dot_product(a, b)
        magnitude_a = VectorOperations.magnitude(a)
        magnitude_b = VectorOperations.magnitude(b)
        angle = math.acos(dot_ab / (magnitude_a * magnitude_b))
        return round(angle * (180.0 / math.pi), 3)

    @staticmethod
    def vector_reflection_plane(a, n):
        projection_on_n = VectorOperations.vector_projection(a, n)
        return VectorOperations.vector_subtraction(a, VectorOperations.scalar_product(projection_on_n, 2))

    @staticmethod
    def vector_projection(a, b):
        dot_ab = VectorOperations.dot_product(a, b)
        dot_bb = VectorOperations.dot_product(b, b)
        scale = dot_ab / dot_bb if dot_bb != 0 else 0
        return VectorOperations.scalar_product(b, scale)

    @staticmethod
    def vector_projection_plane(a, n):
        projection_on_n = VectorOperations.vector_projection(a, n)
        return VectorOperations.vector_subtraction(a, projection_on_n)

    def plot_vector_reflection_plane(a, n):
        reflected_vector = VectorOperations.vector_reflection_plane(a, n)
        projection_on_n = VectorOperations.vector_projection(a, n)
        projection_on_plane = VectorOperations.vector_projection_plane(a, n)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # original vector
        ax.quiver(0, 0, 0, a[0], a[1], a[2], color='r', label='Original Vector')

        # normal vector
        ax.quiver(0, 0, 0, n[0], n[1], n[2], color='g', label='Normal Vector')

        # projection onto the plane
        ax.quiver(0, 0, 0, projection_on_plane[0], projection_on_plane[1], projection_on_plane[2],
                color='b', label='Projection on Plane')

        # reflected vector
        ax.quiver(0, 0, 0, reflected_vector[0], reflected_vector[1], reflected_vector[2],
                color='orange', label='Reflected Vector')

        # Setting the graph limits
        max_range = max(
            abs(coord) for vector in [a, n, reflected_vector, projection_on_plane] for coord in vector
        )
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])

        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.legend()
        plt.show()





    # Vector reflection of a with respect to b
    @staticmethod
    def reflection(a, b):
        scale = 2 * (VectorOperations.dot_product(a, b) / VectorOperations.dot_product(b, b))
        return VectorOperations.vector_subtraction(a, VectorOperations.scalar_product(b, scale))

    def plot_reflection(a, b):
        reflected_vector = VectorOperations.reflection(a, b)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # original vector
        ax.quiver(0, 0, 0, a[0], a[1], a[2], color='r', label='Original Vector')

        # vector b
        ax.quiver(0, 0, 0, b[0], b[1], b[2], color='g', label='Reflection Axis')

        #reflected vector
        ax.quiver(0, 0, 0, reflected_vector[0], reflected_vector[1], reflected_vector[2],
                color='orange', label='Reflected Vector')

        # Setting the graph limits
        max_range = max(
            abs(coord) for vector in [a, b, reflected_vector] for coord in vector
        )
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])

        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.legend()
        plt.show()


    @staticmethod
    def distance_between_vectors(a, b, p=2):
        diff = VectorOperations.vector_subtraction(a, b)
        return VectorOperations.norm(diff, p)

    @staticmethod
    def euclidean_distance(v1, v2):
        if len(v1) != len(v2):
            raise ValueError("Vectors must be of the same dimension.")
        distance = 0
        for i in range(len(v1)):
            distance += (v1[i] - v2[i]) ** 2
        return distance ** 0.5

    @staticmethod
    def manhattan_distance(v1, v2):
        if len(v1) != len(v2):
            raise ValueError("Vectors must be of the same dimension.")
        distance = 0
        for i in range(len(v1)):
            distance += abs(v1[i] - v2[i])
        return distance

    @staticmethod
    def cosine_distance(v1, v2):
        if len(v1) != len(v2):
            raise ValueError("Vectors must be of the same dimension.")
        dot_product = 0
        norm_v1 = 0
        norm_v2 = 0
        for i in range(len(v1)):
            dot_product += v1[i] * v2[i]
            norm_v1 += v1[i] ** 2
            norm_v2 += v2[i] ** 2
        cosine_similarity = dot_product / ((norm_v1 ** 0.5) * (norm_v2 ** 0.5))
        return 1 - cosine_similarity



    # Vector rotation (3D) using Rodrigues' rotation formula
    @staticmethod
    def rotation_3D(a, b, angle):
        b_unit = VectorOperations.unit_vector(b)
        angle_rad = math.radians(angle)
        cos_theta = math.cos(angle_rad)
        sin_theta = math.sin(angle_rad)

        cross_prod = VectorOperations.cross_product(a, b)
        dot_prod = VectorOperations.dot_product(a, b)

        x = VectorOperations.scalar_product(a, cos_theta)
        y = VectorOperations.scalar_product(cross_prod, sin_theta)
        z = VectorOperations.scalar_product(b_unit, dot_prod * (1 - cos_theta))

        return VectorOperations.vector_addition(VectorOperations.vector_addition(x, y), z)

    @staticmethod
    def plot_3D_rotation(a, b, angle):
        rotated_vector = VectorOperations.rotation_3D(a, b, angle)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        #original vector
        ax.quiver(0, 0, 0, a[0], a[1], a[2], color='r', label='Original Vector')

        #rotation axis
        ax.quiver(0, 0, 0, b[0], b[1], b[2], color='g', label='Rotation Axis')

        #rotated vector
        ax.quiver(0, 0, 0, rotated_vector[0], rotated_vector[1], rotated_vector[2], color='b', label=f'Rotated Vector ({angle}°)')

        # Setting the graph limits
        max_range = max(
            abs(coord) for vector in [a, b, rotated_vector] for coord in vector
        )
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])

        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.legend()
        plt.show()



    # Vector-matrix multiplication
    @staticmethod
    def vector_matrix_multiplication(vector, matrix):
        return [sum(vector[i] * matrix[i][j] for i in range(len(vector))) for j in range(len(matrix[0]))]


    @staticmethod
    def linear_interpolation(a, b, t):
        return [(1 - t) * a[i] + t * b[i] for i in range(len(a))]


    # Linear dependence check
    @staticmethod
    def is_linearly_dependent(vectors):
        n = len(vectors)
        m = len(vectors[0])
        matrix = [vec[:] for vec in vectors]

        for i in range(min(n, m)):
            if matrix[i][i] == 0:
                for j in range(i + 1, n):
                    if matrix[j][i] != 0:
                        matrix[i], matrix[j] = matrix[j], matrix[i]
                        break
            for j in range(i + 1, n):
                if matrix[j][i] != 0:
                    ratio = matrix[j][i] / matrix[i][i]
                    for k in range(m):
                        matrix[j][k] -= ratio * matrix[i][k]

        # Check for zero row
        for row in matrix:
            if all(element == 0 for element in row):
                return True
        return False


    # Convolution (1D) of two vectors
    @staticmethod
    def convolution_1d(a, b):
        result = []
        for i in range(len(a) + len(b) - 1):
            sum_conv = 0
            for j in range(len(b)):
                if 0 <= i - j < len(a):
                    sum_conv += a[i - j] * b[j]
            result.append(sum_conv)
        return result

    # Span of vectors check
    @staticmethod
    def is_in_span(vectors, vector):
        augmented_matrix = [vec[:] for vec in vectors]
        augmented_matrix.append(vector[:])

        # Gaussian elimination
        n = len(augmented_matrix)
        m = len(augmented_matrix[0])

        for i in range(min(n, m)):
            if augmented_matrix[i][i] == 0:
                for j in range(i + 1, n):
                    if augmented_matrix[j][i] != 0:
                        augmented_matrix[i], augmented_matrix[j] = augmented_matrix[j], augmented_matrix[i]
                        break
            for j in range(i + 1, n):
                if augmented_matrix[j][i] != 0:
                    ratio = augmented_matrix[j][i] / augmented_matrix[i][i]
                    for k in range(m):
                        augmented_matrix[j][k] -= ratio * augmented_matrix[i][k]

        return all(element == 0 for element in augmented_matrix[-1])

    # Gram-Schmidt Process (Orthogonalization)
    @staticmethod
    def gram_schmidt(vectors):
        orthogonal_basis = []
        for v in vectors:
            for u in orthogonal_basis:
                projection = VectorOperations.vector_projection(v, u)
                v = VectorOperations.vector_subtraction(v, projection)
            if any(component != 0 for component in v):
                orthogonal_basis.append(v)
        return orthogonal_basis


# Example usage
a = [3 ,4 ,5]
b = [1, 0, 0]
n = [0, 0, 1]
c = [7, 8, 9]

print("Scalar Triple Product:", VectorOperations.scalar_triple_product(a, b, c))
print("Vector Triple Product:", VectorOperations.vector_triple_product(a, b, c))
print("Cross Product:", VectorOperations.cross_product(a, b))
print("Magnitude of a:", VectorOperations.magnitude(a))
print("Norm of a (p=3):", VectorOperations.norm(a, 3))
print("Unit Vector of b:", VectorOperations.unit_vector(b))
print("Vector Subtraction:", VectorOperations.vector_subtraction(a, b))
print("Vector Addition:", VectorOperations.vector_addition(a, b))
print("Scalar Product (a * 3):", VectorOperations.scalar_product(a, 3))
print("Dot Product:", VectorOperations.dot_product(a, b))
print("Hadamard Product:", VectorOperations.hadamard_product(a, b))
print("Outer Product:", VectorOperations.outer_product(a, b))
print("Linear Combination ([a, b], [2, 3]):", VectorOperations.linear_combination([a, b], [2, 3]))
print("Angle Between Vectors:", VectorOperations.angle_between_vectors(a, b))
print("Projection of a onto b:", VectorOperations.vector_projection(a, b))
print("Projection of a onto plane with normal n:", VectorOperations.plot_vector_reflection_plane(a,n))
print("Reflection of a with respect to b:", VectorOperations.plot_reflection(a, b))
print("Distance Between a and b (p=2):", VectorOperations.distance_between_vectors(a, b))
print("Rotation of a around b by 90 degrees:", VectorOperations.plot_3D_rotation(a, b, 90))

print("Are vectors linearly dependent?", VectorOperations.is_linearly_dependent([a, b, n]))
print("Is vector in span:", VectorOperations.is_in_span([a, b], [4, 5, 6]))
print("Gram-Schmidt Orthogonal Basis:", VectorOperations.gram_schmidt([a, b, n]))
print("Convolution 1D:", VectorOperations.convolution_1d(a, b))
print("Linear Interpolation (t=0.5):", VectorOperations.linear_interpolation(a, b, 0.5))

v1 = [1, 2, 3]
v2 = [4, 5, 6]

print("Euclidean Distance:", VectorOperations.euclidean_distance(v1, v2))
print("Manhattan Distance:", VectorOperations.manhattan_distance(v1, v2))
print("Cosine Distance:",VectorOperations.cosine_distance(v1, v2))

