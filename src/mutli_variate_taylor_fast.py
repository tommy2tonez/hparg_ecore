from typing import Callable
import copy
import math
import functools
import random
from typing import Protocol
import sys

#new operations

#2sum
#3sum
#4sum

#

def get_slope(f: Callable[[float], float], x: int, derivative_order: int, a: float = 0.000001) -> float:

    if derivative_order == 0:
        return f(x)

    return (get_slope(f, x + a, derivative_order - 1) - get_slope(f, x, derivative_order - 1)) / a  

def newton_approx(operation: Callable[[float], float], iteration_sz: int, initial_x: float, a: float = 0.001) -> tuple[float, float]:

    #this is not correctly implemented
    cur_x       = initial_x
    min_y       = abs(operation(cur_x))
    cand_x      = cur_x 
    epsilon     = float(0.01)

    for _ in range(iteration_sz):
        cur_y   = operation(cur_x)

        if (abs(cur_y) < min_y):
            cand_x  = cur_x
            min_y   = cur_y

        a_y     = operation(cur_x + a)
        slope   = (a_y - cur_y) / a

        if (abs(slope) < epsilon):
            break 

        cur_x   -= cur_y / slope

    return cand_x, min_y

def tom_approx(operation: Callable[[float], float], iteration_sz: int, initial_x: float, a: float = 0.001) -> tuple[float, float]:

    s0      = get_slope(operation, initial_x, 0)
    v       = get_slope(operation, initial_x, 1)
    accel   = get_slope(operation, initial_x, 2)
    
    epsilon = float(0.01)
    a       = 1/2 * accel
    b       = v
    c       = s0

    if abs(a) < epsilon:
        return newton_approx(operation, iteration_sz, initial_x)

    delta   = b ** 2 - 4 * a * c

    if delta > 0:
        x1  = (-b + math.sqrt(delta)) / (2*a)
        x2  = (-b - math.sqrt(delta)) / (2*a)

        (x1, y1)    = newton_approx(operation, iteration_sz, x1, a)
        (x2, y2)    = newton_approx(operation, iteration_sz, x2, a)

        if abs(y1) < abs(y2):
            return x1, y1

        return x2, y2

    x = -b / (2*a)

    return newton_approx(operation, iteration_sz, x, a)

def tom_approx2(operation: Callable[[float], float], iteration_sz: int, initial_x: float, a: float = 0.001) -> tuple[float, float]:

    NEWTON_ITER_SZ: int = 3 

    if iteration_sz == 0:
        return newton_approx(operation, NEWTON_ITER_SZ, initial_x, a)

    s0      = get_slope(operation, initial_x, 0)
    v       = get_slope(operation, initial_x, 1)
    accel   = get_slope(operation, initial_x, 2)
    
    epsilon = float(0.01)
    a       = float(1)/2 * accel
    b       = v
    c       = s0

    if abs(a) < epsilon:
        return newton_approx(operation, NEWTON_ITER_SZ, initial_x)

    delta   = b ** 2 - 4 * a * c

    if delta > 0:
        x1  = (-b + math.sqrt(delta)) / (2*a)
        x2  = (-b - math.sqrt(delta)) / (2*a)

        (x1, y1)    = tom_approx2(operation, iteration_sz - 1, x1, a)
        (x2, y2)    = tom_approx2(operation, iteration_sz - 1, x2, a)

        if abs(y1) < abs(y2) and abs(y1) < abs(s0):
            return x1, y1
        
        if abs(y2) < abs(s0):
            return x2, y2
        
        return initial_x, s0

    x           = -b / (2*a)
    (x1, y1)    = tom_approx2(operation, iteration_sz - 1, x, a)

    if abs(y1) < abs(s0):
        return x1, y1

    return initial_x, s0

def stable_approx(operation: Callable[[float], float], iteration_sz: int, initial_x: float, a: float = 0.001) -> tuple[float, float]:

    return tom_approx2(operation, 3, initial_x, a)

def scalar_multiply_vector(c : float, arr: list[float]) -> list[float]:

    return [c * e for e in arr]

def add_vector(lhs: list[float], rhs: list[float]) -> list[float]:

    return [e_lhs + e_rhs for (e_lhs, e_rhs) in zip(lhs, rhs)] 

def sub_vector(lhs: list[float], rhs: list[float]) -> list[float]:

    return [e - e1 for (e, e1) in zip(lhs, rhs)] 

def dot_product(lhs: list[float], rhs: list[float]) -> float:

    return sum([e * e1 for (e, e1) in zip(lhs, rhs)]) 

def to_scalar_value(vector: list[float]) -> float:

    return math.sqrt(dot_product(vector, vector)) 

def exponential_randomize_frange(frange: float, exp_base: float) -> float:

    exp_first: float    = float(0) #0 for now
    exp_last: float     = math.log(frange, exp_base)
    exp_diff: float     = exp_last - exp_first
    exp_rand: float     = random.random() * exp_diff

    return exp_rand + exp_first

def uniform_randomize_frange(frange: float) -> float:

    return frange * random.random()

def split_vector(vec: list[float], lhs_sz: int) -> tuple[list[float], list[float]]:
    
    lhs: list[float]    = vec[:lhs_sz]
    rhs: list[float]    = vec[lhs_sz:]
    
    return (lhs, rhs)

def random_0(sz: int) -> float:

    dice = random.randrange(0, sz)
    
    if dice == 0:
        return float(0)
    
    return float(1)

def random_sign(sz: int) -> float:

    dice = random.randrange(0, sz)

    if dice == 0:
        return float(-1)

    return float(1) 

def flip_a_coin() -> bool:

    return bool(random.randrange(0, 2)) 

def get_random_vector(dimension_sz: int) -> list[float]:

    return [random.random() * random_0(2) * random_sign(2) for _ in range(dimension_sz)]

def get_unit_vector(vector: list[float]) -> list[float]:

    sz: float = to_scalar_value(vector)
    return scalar_multiply_vector(float(1) / sz, vector)

def rand_multidimensional_sphere_radian(dimension_sz: int) -> list[float]:

    return [2 * math.pi * random.random() for _ in range(dimension_sz)] 

def radian_coordinate_to_euclidean_coordinate(coor: list[float]) -> list[float]:

    if len(coor) == 0:
        return []

    if len(coor) == 1:
        return [float(1)]

    sliced_coor: list[float] = radian_coordinate_to_euclidean_coordinate(coor[1:])

    return [math.cos(coor[0])] + [math.sin(coor[0]) * sliced_coor[i] for i in range(len(sliced_coor))] 

def pairwise_multiply_vector(lhs: list[float], rhs: list[float]):

    return [lhs[i] * rhs[i] for i in range(len(lhs))] 

def avg_invsqr_list(a: list[float]) -> float:

    if len(a) == 0:
        return float(0)

    return float(1) / (sum([float(1) / (a[i] ** 4) for i in range(len(a))]) / len(a)) #what's happening? sum(1 / x^2) would approximate the global maxima - which represents the anomaly or our <looking_for> value - 1/those would turn things upside down (minima <-> maxima) - we are looking for global minima - which is the criteria for newton_approx  

def get_no_jagged_shape(a: object) -> list[int]:

    if type(a) != type([]):
        return [] 

    if len(a) == 0:
        return [0]

    return [len(a)] + get_no_jagged_shape(a[0])

def flatten(obj: object) -> object:

    if type(obj) == type(list()):
        rs: list = []

        for e in obj:
            rs += flatten(e)

        return rs

    return [obj]

def get_space_size(space_vec: list[int]) -> int:

    if len(space_vec) == 0:
        return 0

    return functools.reduce(lambda x, y: x * y, space_vec, 1)

def shape_as(lhs: list, shape: list[int]) -> object:

    flattend_lhs: list  = flatten(lhs)
    space_sz: int       = get_space_size(shape)

    if len(flattend_lhs) != space_sz:
        raise Exception()

    if len(shape) == 1:
        return lhs

    sub_space_sz: int   = int(space_sz / shape[0])
    offset: int         = 0
    rs: list            = []

    for i in range(shape[0]):
        sliced: list    = lhs[offset: offset + sub_space_sz]
        sub_space: list = shape_as(sliced, shape[1:])
        rs              += [sub_space]
        offset          += sub_space_sz

    return rs

class Coordinate:

    def __init__(self, coor_value: list[float]):

        self.coor_value = copy.deepcopy(coor_value)

    def size(self) -> int:

        return len(self.coor_value) 

    def at(self, idx: int) -> float:

        return self.coor_value[idx] 

    def raw(self) -> list[float]:

        return copy.deepcopy(self.coor_value) 

class StepperInterface(Protocol):

    def step(self) -> float:
        ...

    def has_next_step(self) -> bool:
        ... 

    def reset(self):
        ...

class ExponentialStepper:

    def __init__(self, y0: float, exp_base: float, exp_step_sz: int): #uint

        self.y0             = y0
        self.exp_base       = exp_base
        self.exp_step_sz    = exp_step_sz
        self.cur_step_idx   = 0
    
    def step(self) -> float:

        rs                  = self.y0 + math.pow(self.exp_base, self.cur_step_idx)
        self.cur_step_idx   = min(self.exp_step_sz, self.cur_step_idx + 1)

        return rs

    def has_next_step(self) -> bool:

        return self.cur_step_idx < self.exp_step_sz

    def reset(self):

        self.cur_step_idx = 0

class LinearStepper: 

    def __init__(self, y0: float, a: float, step_sz: int):

        self.y0             = y0
        self.a              = a
        self.step_sz        = step_sz
        self.cur_step_idx   = 0

    def step(self) -> float:

        rs                  = self.y0 + self.a * self.cur_step_idx
        self.cur_step_idx   = min(self.step_sz, self.cur_step_idx + 1)

        return rs

    def has_next_step(self) -> bool:

        return self.cur_step_idx < self.step_sz

    def reset(self):

        self.cur_step_idx = 0 

class RandomizerInterface(Protocol):

    def randomize(self) -> float:
        ...

class ExponentialRandomizerInterface:

    def __init__(self, y0: float, frange: float, exp_base: float):
        
        self.y0         = y0
        self.frange     = frange
        self.exp_base   = exp_base

    def randomize(self) -> float:
        
        return exponential_randomize_frange(self.frange, self.exp_base) + self.y0

class UniformRandomizerInterface:

    def __init__(self, y0: float, frange: float):

        self.y0         = y0
        self.frange     = frange

    def randomize(self) -> float:
        
        return uniform_randomize_frange(self.frange) + self.y0

class BallisticDeviceInterface(Protocol):

    def shoot(self, t: float) -> list[Coordinate]:
        ...

class BulletBallisticDevice:

    def __init__(self, coor: Coordinate):

        self.coor = Coordinate(coor.raw()) 

    def shoot(self, t: float) -> list[Coordinate]:

        raw_coor: list[float]       = self.coor.raw()
        scaled_coor: list[float]    = scalar_multiply_vector(t, raw_coor)

        return [Coordinate(scaled_coor)]        

class SphereMagneticBallisticDevice:

    def __init__(self, s0_rad_coor: Coordinate, direction_vec: Coordinate, frequency_coeff: float, r: float):

        self.s0_rad_coor        = Coordinate(s0_rad_coor.raw())
        self.direction_vec      = Coordinate(direction_vec.raw())
        self.frequency_coeff    = frequency_coeff
        self.r                  = r

    def shoot(self, t: float) -> list[Coordinate]:

        raw_s0_rad_coor: list[float]            = self.s0_rad_coor.raw()
        raw_dir_rad_coor: list[float]           = self.direction_vec.raw()

        bullet_rad_coor: list[float]            = add_vector(raw_s0_rad_coor, scalar_multiply_vector(self.frequency_coeff * t, raw_dir_rad_coor))
        bullet_euclid_coor: list[float]         = radian_coordinate_to_euclidean_coordinate(bullet_rad_coor)
        scaled_bullet_euclid_coor: list[float]  = scalar_multiply_vector(self.r, bullet_euclid_coor)

        return [Coordinate(scaled_bullet_euclid_coor)]

class RandomSphereMagneticBallisticDevice(SphereMagneticBallisticDevice):

    def __init__(self, dimension_sz: int, max_r: float, max_frequency_coeff: float):

        s0_rad_coor: list[float]        = rand_multidimensional_sphere_radian(dimension_sz)
        directional_vec: list[float]    = get_random_vector(dimension_sz)
        r: float                        = uniform_randomize_frange(max_r)
        frequency_coeff: float          = uniform_randomize_frange(max_frequency_coeff)

        super().__init__(Coordinate(s0_rad_coor), Coordinate(directional_vec), frequency_coeff, r)

class SpheroidMagneticBallisticDevice:

    def __init__(self, s0_rad_coor: Coordinate, direction_vec: Coordinate, frequency_coeff: float, oval_shape: Coordinate):

        self.s0_rad_coor        = Coordinate(s0_rad_coor.raw())
        self.direction_vec      = Coordinate(direction_vec.raw())
        self.frequency_coeff    = frequency_coeff
        self.oval_shape         = Coordinate(oval_shape.raw()) 

    def shoot(self, t: float) -> list[Coordinate]:

        raw_s0_rad_coor: list[float]            = self.s0_rad_coor.raw()
        raw_dir_rad_coor: list[float]           = self.direction_vec.raw()

        bullet_rad_coor: list[float]            = add_vector(raw_s0_rad_coor, scalar_multiply_vector(self.frequency_coeff * t, raw_dir_rad_coor))
        bullet_euclid_coor: list[float]         = radian_coordinate_to_euclidean_coordinate(bullet_rad_coor)
        scaled_bullet_euclid_coor: list[float]  = pairwise_multiply_vector(self.oval_shape.raw(), bullet_euclid_coor)

        return [Coordinate(scaled_bullet_euclid_coor)]

class RandomSphroidMagneticBallisticDevice(SpheroidMagneticBallisticDevice):

    def __init__(self, dimension_sz: int, max_r: float, max_frequency_coeff: float):

        s0_rad_coor: list[float]        = rand_multidimensional_sphere_radian(dimension_sz)
        directional_vec: list[float]    = get_random_vector(dimension_sz)
        r: float                        = uniform_randomize_frange(max_r)
        frequency_coeff: float          = uniform_randomize_frange(max_frequency_coeff)
        oval_shape: list[float]         = scalar_multiply_vector(r, get_random_vector(dimension_sz)) 

        super().__init__(Coordinate(s0_rad_coor), Coordinate(directional_vec), frequency_coeff, Coordinate(oval_shape)) 

class RandomTwoArmsBallisticDevice:

    def __init__(self, dimension_sz: int, 
                 max_arm1_length: float, max_arm1_frequency_coeff: float, 
                 max_arm2_length: float, max_arm2_frequency_coeff: float):

        self.rotating_arm1: BallisticDeviceInterface  = RandomSphereMagneticBallisticDevice(dimension_sz, max_arm1_length, max_arm1_frequency_coeff)
        self.rotating_arm2: BallisticDeviceInterface  = RandomSphereMagneticBallisticDevice(dimension_sz, max_arm2_length, max_arm2_frequency_coeff)

    def shoot(self, t: float) -> list[Coordinate]:

        arm1_coor: Coordinate = self.rotating_arm1.shoot(t)[0]
        arm2_coor: Coordinate = self.rotating_arm2.shoot(t)[0]

        return [Coordinate(add_vector(arm1_coor.raw(), arm2_coor.raw()))]

class RandomThreeArmBallisticDevice:

    def __init__(self, dimension_sz: int, 
                 max_arm1_length: float, max_arm1_frequency_coeff: float, 
                 max_arm2_length: float, max_arm2_frequency_coeff: float, 
                 max_arm3_length: float, max_arm3_frequency_coeff: float):

        self.rotating_arm1: BallisticDeviceInterface = RandomSphroidMagneticBallisticDevice(dimension_sz, max_arm1_length, max_arm1_frequency_coeff)
        self.rotating_arm2: BallisticDeviceInterface = RandomSphroidMagneticBallisticDevice(dimension_sz, max_arm2_length, max_arm2_frequency_coeff)
        self.rotating_arm3: BallisticDeviceInterface = RandomSphroidMagneticBallisticDevice(dimension_sz, max_arm3_length, max_arm3_frequency_coeff)

    def shoot(self, t: float) -> list[Coordinate]:

        arm1_coor: Coordinate = self.rotating_arm1.shoot(t)[0]
        arm2_coor: Coordinate = self.rotating_arm2.shoot(t)[0]
        arm3_coor: Coordinate = self.rotating_arm3.shoot(t)[0]

        return [Coordinate(add_vector(add_vector(arm1_coor.raw(), arm2_coor.raw()), arm3_coor.raw()))] 

class StaticPointBagBallisticDevice:

    def __init__(self, point_bag: list[Coordinate]):

        self.point_bag = [Coordinate(point.raw()) for point in point_bag]

    def shoot(self, t: float) -> list[Coordinate]:

        return [Coordinate(e.raw()) for e in self.point_bag]

class UniformRandomStaticPointBagBallisticDevice(StaticPointBagBallisticDevice):

    def __init__(self, dimension_sz: int, bag_sz: int, y_range: float):

        point_bag: list[Coordinate] = [Coordinate(scalar_multiply_vector(y_range, get_random_vector(dimension_sz))) for _ in range(bag_sz)] 
        super().__init__(point_bag)

class UniformRandomStaticSpherePointBagBallisticDevice(StaticPointBagBallisticDevice):

    def __init__(self, dimension_sz: int, bag_sz: int, r: float):

        point_bag: list[Coordinate] = [Coordinate(scalar_multiply_vector(r, radian_coordinate_to_euclidean_coordinate(rand_multidimensional_sphere_radian(dimension_sz)))) for _ in range(bag_sz)] 
        super().__init__(point_bag)

class CircumscribingStaticPointBagBallisticDevice:

    def __init__(self, point_bag: list[Coordinate], frequency_coeff: float):

        self.ballistic          = StaticPointBagBallisticDevice(point_bag)
        self.frequency_coeff    = frequency_coeff

    def shoot(self, t: float) -> list[Coordinate]:

        point_bag: list[Coordinate] = self.ballistic.shoot(t)
        return [Coordinate(scalar_multiply_vector(self.frequency_coeff * t, point.raw())) for point in point_bag] 

class ChainedBallisticDevice:

    def __init__(self, ballistic_device_arr: list[BallisticDeviceInterface]):

        self.ballistic_device_arr: list[BallisticDeviceInterface] = ballistic_device_arr

    def shoot(self, t: float) -> list[Coordinate]:

        rs: list[list[Coordinate]] = []

        for i in range(len(self.ballistic_device_arr)):
            rs += [self.ballistic_device_arr[i].shoot(t)]

        return self._combinatorial_reduce_add(rs)

    def _combinatorial_reduce_add(self, inp: list[list[Coordinate]]) -> list[Coordinate]:

        if len(inp) == 0:
            return []

        def _internal_reduce(lhs: list[Coordinate], rhs: list[Coordinate]) -> list[Coordinate]:

            rs: list[Coordinate] = []

            for e_lhs in lhs:
                for e_rhs in rhs:
                    e_rs    = add_vector(e_lhs.raw(), e_rhs.raw()) 
                    rs      += [Coordinate(e_rs)]

            return rs

        return functools.reduce(_internal_reduce, inp[1:], inp[0])

def get_random_twoarms_ballistic_device(dimension_sz: int, _range: float) -> BallisticDeviceInterface:

    arm1_frequency_coeff: float = (random.random() * 10) ** random.randrange(10)
    arm2_frequency_coeff: float = (random.random() * 10) ** random.randrange(10)

    return RandomTwoArmsBallisticDevice(dimension_sz, 
                                        _range, arm1_frequency_coeff,
                                        _range, arm2_frequency_coeff)

def get_random_threearms_ballistic_device(dimension_sz: int, _range: float) -> BallisticDeviceInterface:

    arm1_frequency_coeff: float = (random.random() * 10) ** random.randrange(10)
    arm2_frequency_coeff: float = (random.random() * 10) ** random.randrange(10)
    arm3_frequency_coeff: float = (random.random() * 10) ** random.randrange(10)

    return RandomThreeArmBallisticDevice(dimension_sz, 
                                         _range, arm1_frequency_coeff, 
                                         _range, arm2_frequency_coeff,
                                         _range, arm3_frequency_coeff)

def get_random_circumscribing_ballistic_device(dimension_sz: int, _range: float, bag_sz_range: int = 16, bag_sz_min: int = 1) -> BallisticDeviceInterface:

    bag_sz: int                 = min(random.randrange(bag_sz_range), bag_sz_min)
    radius: int                 = uniform_randomize_frange(_range)
    frequency_coeff: float      = (random.random() * 10) ** random.randrange(10)
    point_bag: list[Coordinate] = [Coordinate(scalar_multiply_vector(radius, radian_coordinate_to_euclidean_coordinate(rand_multidimensional_sphere_radian(dimension_sz)))) for _ in range(bag_sz)] 

    return CircumscribingStaticPointBagBallisticDevice(point_bag, frequency_coeff)

def get_random_point_bag_ballistic_device(dimension_sz: int, _range: float, bag_sz_range: int = 16, bag_sz_min: int = 1) -> BallisticDeviceInterface:

    bag_sz: int = min(random.randrange(bag_sz_range), bag_sz_min)
    radius: int = uniform_randomize_frange(_range)

    return UniformRandomStaticPointBagBallisticDevice(dimension_sz, bag_sz, radius)

def get_random_melee_ballistic_device(dimension_sz: int, _range: float) -> BallisticDeviceInterface:

    device_sz_range: int    = 3
    device_sz_min: int      = 1
    device_sz: int          = max(random.randrange(device_sz_range), device_sz_min)

    device_list: list[BallisticDeviceInterface]                                     = []
    random_melee_device_gen: list[Callable[[int, float], BallisticDeviceInterface]] = [get_random_twoarms_ballistic_device, get_random_threearms_ballistic_device, get_random_circumscribing_ballistic_device, get_random_point_bag_ballistic_device]

    for _ in range(device_sz):
        idx = random.randrange(0, len(random_melee_device_gen))
        device: BallisticDeviceInterface = random_melee_device_gen[idx](dimension_sz, _range)
        device_list += [device]

    return ChainedBallisticDevice(device_list) 

def get_random_bullet_ballistic_device(dimension_sz: int) -> BallisticDeviceInterface:
    
    return BulletBallisticDevice(Coordinate(get_random_vector(dimension_sz)))

def get_random_spheremagnetic_ballistic_device(dimension_sz: int, _range: float) -> BallisticDeviceInterface:
    
    frequency_coeff: float = (random.random() * 10) ** random.randrange(10)
    return RandomSphereMagneticBallisticDevice(dimension_sz, _range, frequency_coeff)

def get_random_spheroidmagnetic_ballistic_device(dimension_sz: int, _range: float) -> BallisticDeviceInterface:

    frequency_coeff: float = (random.random() * 10) ** random.randrange(10)
    return RandomSphroidMagneticBallisticDevice(dimension_sz, _range, frequency_coeff) 

def get_random_range_ballistic_device(dimension_sz: int, magnetic_radius: float) -> BallisticDeviceInterface:
    
    infinite_ballistic_gen: list    = [get_random_bullet_ballistic_device]
    finite_ballistic_gen: list      = [get_random_spheremagnetic_ballistic_device, get_random_spheroidmagnetic_ballistic_device]

    inifite_device: BallisticDeviceInterface    = infinite_ballistic_gen[random.randrange(0, len(infinite_ballistic_gen))](dimension_sz)
    finite_device: BallisticDeviceInterface     = finite_ballistic_gen[random.randrange(0, len(finite_ballistic_gen))](dimension_sz, magnetic_radius)
 
    return ChainedBallisticDevice([inifite_device, finite_device])

def get_random_rangemelee_ballistic_device(dimension_sz: int, magnetic_radius: float, melee_range: float) -> BallisticDeviceInterface:

    range_device: BallisticDeviceInterface = get_random_range_ballistic_device(dimension_sz, magnetic_radius)
    melee_device: BallisticDeviceInterface = get_random_melee_ballistic_device(dimension_sz, melee_range)

    return ChainedBallisticDevice([range_device, melee_device])

def get_random_ballistic_device(dimension_sz: int) -> BallisticDeviceInterface:

    magnetic_radius: float  = (random.random() * 10) ** random.randrange(0, 10)
    melee_range: float      = (random.random() * 10) ** random.randrange(0, 10)
    rand_value: int         = random.randrange(0, 3)

    if rand_value == 0:
        return get_random_melee_ballistic_device(dimension_sz, melee_range)

    if rand_value == 1:
        return get_random_range_ballistic_device(dimension_sz, magnetic_radius)

    return get_random_rangemelee_ballistic_device(dimension_sz, magnetic_radius, melee_range)  

class NewtonOptimizerInterface(Protocol):

    def optimize(self, f: Callable[[float], float], x0: float) -> float:
        ...

class TwoOrderStepNewtonOptimizer:

    def __init__(self, stepper: StepperInterface, iteration_sz: int = 4, a: float = 0.001):

        self.stepper        = stepper
        self.iteration_sz   = iteration_sz
        self.a              = a

    def optimize(self, f: Callable[[float], float], x0: float) -> float:

        x_cursor: float                         = x0
        x_cand_list: list[tuple[float, float]]  = [(f(x0), x0)]

        while self.stepper.has_next_step():
            x_cursor    = x_cursor + self.stepper.step()
            (x_cand, y) = tom_approx2(f, self.iteration_sz, x_cursor, self.a)
            x_cand_list += [(abs(y), x_cand)]

        self.stepper.reset()
    
        return min(x_cand_list)[1]  

class URLinearTwoOrderNewtonOptimizer(TwoOrderStepNewtonOptimizer):

    def __init__(self, y0_first: float = float(0), y0_last: float = float(0), 
                 a_abs_range: float = float(10), a_abs_range_min: float = 0.001,
                 step_sz_range: int = 32, step_sz_min: int = 1,
                 iteration_sz: int = 4, derivative_offset: float = 0.001):

        y0: float       = uniform_randomize_frange(y0_last - y0_first) + y0_first
        signness: float = float(-1) if flip_a_coin() else float(1)
        a: float        = max(uniform_randomize_frange(a_abs_range), a_abs_range_min) * signness 
        step_sz: int    = max(step_sz_min, random.randrange(0, step_sz_range))

        super().__init__(LinearStepper(y0, a, step_sz), iteration_sz, derivative_offset)

class ERTwoOrderNewtonOptimizer(TwoOrderStepNewtonOptimizer):

    def __init__(self, y0_first: float = float(0), y0_last: float = float(0),
                 exp_base_range: float = float(10), exp_base_range_min: float = 1,
                 exp_step_range: int = 10, exp_step_min: int = 1,
                 iteration_sz: int = 4, derivative_offset: float = 0.001):

        y0: float       = uniform_randomize_frange(y0_last - y0_first) + y0_first
        exp_base: float = max(uniform_randomize_frange(exp_base_range), exp_base_range_min)
        exp_step: int   = max(random.randrange(0, exp_step_range), exp_step_min)

        super().__init__(ExponentialStepper(y0, exp_base, exp_step), iteration_sz, derivative_offset)

def get_random_linear_twoorder_newton_optimizer() -> NewtonOptimizerInterface:

    a_abs_range: float = (random.random() * 10) ** random.randrange(0, 10)
    return URLinearTwoOrderNewtonOptimizer(0, 0, a_abs_range)

def get_random_exponential_twoorder_newton_optimizer() -> NewtonOptimizerInterface:

    return ERTwoOrderNewtonOptimizer() 

def get_random_twoorder_newton_optimizer() -> NewtonOptimizerInterface:

    if flip_a_coin():
        return get_random_linear_twoorder_newton_optimizer()
    else:
        return get_random_exponential_twoorder_newton_optimizer() 

class TaylorApprox:

    def __init__(self, node_list: list[object] = None, coeff_value_arr: list[list[float]] = None):

        self.node_list          = node_list
        self.coeff_value_arr    = coeff_value_arr

    def approx(self, x: list[float]) -> list[float]:

        if self.node_list == None and self.coeff_value_arr == None:
            return []

        vec_arr: list[list[float]] = []

        if self.coeff_value_arr != None:
            for i in range(len(self.coeff_value_arr)):
                factorial_multiplier: float = 1 / math.factorial(i)
                coeff: list[float]          = self.coeff_value_arr[i] 
                x_multiplier                = x[-1] ** i
                scaled_coeff: list[float]   = scalar_multiply_vector(x_multiplier * factorial_multiplier, coeff)
                vec_arr                     += [scaled_coeff] 
        else:    
            for i in range(len(self.node_list)):
                factorial_multiplier: float = 1 / math.factorial(i)
                coeff: list[float]          = self.node_list[i].approx(x[:-1])
                x_multiplier                = x[-1] ** i
                scaled_coeff: list[float]   = scalar_multiply_vector(x_multiplier * factorial_multiplier, coeff)
                vec_arr                     += [scaled_coeff]

        return functools.reduce(add_vector, vec_arr)

    def clone(self) -> object:

        rs: TaylorApprox = TaylorApprox()

        if self.node_list != None:
            rs.node_list = [node.clone() for node in self.node_list]

        if self.coeff_value_arr != None:
            rs.coeff_value_arr = copy.deepcopy(self.coeff_value_arr)

        return rs

    def coeff_size(self) -> int:

        sz: int = 0

        if self.node_list != None:
            for node in self.node_list:
                sz += node.coeff_size()

        if self.coeff_value_arr != None:
            sz += len(self.coeff_value_arr)

        return sz

    def coeff_dump(self) -> list[list[float]]:

        rs: list[list[float]] = []

        if self.node_list != None:
            for node in self.node_list:
                node_list_coeff: list[list[float]] = node.coeff_dump()
                rs += node_list_coeff 

        if self.coeff_value_arr != None:
            rs += self.coeff_value_arr

        return rs

    def coeff_load(self, arr: list[list[float]]):

        offset: int = 0

        if self.node_list != None:
            for node in self.node_list:
                sliced_sz: int              = node.coeff_size()
                sliced: list[list[float]]   = arr[offset: offset + sliced_sz]
                node.coeff_load(sliced)
                offset                      += sliced_sz
        
        if self.coeff_value_arr != None:
            self.coeff_value_arr = copy.deepcopy(arr[offset:])

class CallableTaylorApprox:

    def __init__(self, taylor_approx: TaylorApprox):

        self.taylor_approx = taylor_approx

    def __call__(self, x: list[float]) -> list[float]:

        return self.taylor_approx.approx(x)

class DeviationCalculatorInterface(Protocol):

    def deviation(self, f: Callable[[list[float]], list[float]], instrument: Callable[[list[float]], list[float]]) -> float:
        ... 

def kw_pow(x: object, y: object) -> object:

    return math.pow(x, y)

class DeviationCalculator:

    def __init__(self, point_arr: list[list[float]]):

        self.point_arr = copy.deepcopy(point_arr)

    def deviation(self, approxer: Callable[[list[float]], list[float]], instrument: Callable[[list[float]], list[float]]) -> float:

        distance_vec: list[list[float]] = []

        for point in self.point_arr:
            projected_a: list[float]    = approxer(point)
            projected_i: list[float]    = instrument(point)
            distance_vec                += [sub_vector(projected_a, projected_i)]

        scalar_distance_vec: list[float]    = list(map(to_scalar_value, distance_vec))
        sqr_distance_vec: list[float]       = list(map(functools.partial(kw_pow, y = float(2)), scalar_distance_vec))

        if len(sqr_distance_vec) == 0:
            return float(0) 

        return math.sqrt(sum(sqr_distance_vec) / len(sqr_distance_vec))

class RandomDiscreteDeviationCalculator(DeviationCalculator):

    def __init__(self, coor_range: list[float], sz: int):

        point_arr: list[list[float]] = [list(map(uniform_randomize_frange, coor_range)) for _ in range(sz)]
        super().__init__(point_arr)

def make_2d_arr(x_sz: int, y_sz: int, initial_value: object):

    return [[copy.deepcopy(initial_value) for __ in range(y_sz)] for _ in range(x_sz)]

#I was proving the completeness + most compactness of this approximation in terms of <there exists no better representation, only equivalent representations>
#it's hard to prove, in order to do so, we must infer that s, v, a, j, ... are not logically-tangled in terms of entropy
#we'll port this code to C for quant proof of concept tomorrow, it's gonna be a bumpy low-level code

#alright, the only implementable optimization we could do for our current model is directional optimization 
#we have a crazy amount of output, we have the <magnetic_direction> for all of the logits, we find the <gradient_update> benefits, and we maxwell those 

#the problem is: how could we possibly pass such information backwardly or we pass it forwardly?
#unless we do random sampling of directions for backwarding tiles

#alright consider this flow
#we randomize the <projectile> or the cursor, we allocate computing tiles out of leafs inching in the direction, we compute the results, we crit the results, we maxwell the results (uacm + pacm), and we backprop those guys to the leafs 
#what about our Newton approx? its the msgr_forward responsibility, we compute the deviation projection externally + add to the compute queue later on 
#point is we have tons of compute, yet not a parallel algorithm to do this
#client is asking for $BB if we could get this correctly from A to Z
#there is no just in time or fancy accelerated linear, etc.
#like Agent Smith once told, we multiply
#alright fellas, embrace Taylor Swift, not delusions
#we have so many variables to compute from virtual machine snapshots that our AI's gonna know literally everything, and we could read our Sunday newspaper through our AI

def make_taylor_model(in_variable_sz: int, out_variable_sz: int, derivative_order_sz: int) -> TaylorApprox:

    if in_variable_sz == 0:
        return TaylorApprox([])

    if in_variable_sz == 1:
        return TaylorApprox([], make_2d_arr(derivative_order_sz, out_variable_sz, float(0)))

    node_rs: list[TaylorApprox] = []

    for _ in range(derivative_order_sz):
        f_x: TaylorApprox   = make_taylor_model(in_variable_sz - 1, out_variable_sz, derivative_order_sz) 
        node_rs             += [f_x]

    return TaylorApprox(node_rs)

def ballistic_optimize(taylor_coeff: list[float], 
                       ballistic_device: BallisticDeviceInterface,
                       coeff_functionizer: Callable[[list[float]], Callable[[list[float]], list[float]]],
                       instrument: Callable[[list[float]], list[float]],
                       deviation_calculator: DeviationCalculatorInterface) -> tuple[list[float], float]:

    optimizer: NewtonOptimizerInterface = get_random_twoorder_newton_optimizer()
    x0: float                           = float(0)

    def _deviation_negotiator(t: float) -> float:
        ballistic_coor_list: list[Coordinate]   = ballistic_device.shoot(t)
        calibrated_coor_list: list[Coordinate]  = [Coordinate(add_vector(taylor_coeff, coor.raw())) for coor in ballistic_coor_list] 
        deviation_list: list[float]             = []

        for taylor_model_coor in calibrated_coor_list:
            taylor_function     = coeff_functionizer(taylor_model_coor.raw())
            deviation: float    = deviation_calculator.deviation(taylor_function, instrument)
            deviation_list      += [deviation]

        return avg_invsqr_list(deviation_list)

    x: float                                = optimizer.optimize(_deviation_negotiator, x0)
    ballistic_coor_list: list[Coordinate]   = ballistic_device.shoot(x)
    calibrated_coor_list: list[Coordinate]  = [Coordinate(add_vector(taylor_coeff, coor.raw())) for coor in ballistic_coor_list]
    deviation_list: list[float]             = []

    for taylor_model_coor in calibrated_coor_list:
        taylor_function     = coeff_functionizer(taylor_model_coor.raw())
        deviation: float    = deviation_calculator.deviation(taylor_function, instrument)
        deviation_list      += [(deviation, taylor_model_coor.raw())]

    if len(deviation_list) == 0:
        return (taylor_coeff, sys.float_info.max)

    return (min(deviation_list)[1], min(deviation_list)[0])

def range_ballistic_optimize(taylor_coeff: list[float],
                             coeff_functionizer: Callable[[list[float]], Callable[[list[float]], list[float]]],
                             instrument: Callable[[list[float]], list[float]],
                             deviation_calculator: DeviationCalculatorInterface,
                             magnetic_range: float = float(16)):

    return ballistic_optimize(taylor_coeff, 
                              get_random_range_ballistic_device(len(taylor_coeff), magnetic_range), 
                              coeff_functionizer,
                              instrument,
                              deviation_calculator)

def melee_ballistic_optimize(taylor_coeff: list[float],
                             coeff_functionizer: Callable[[list[float]], Callable[[list[float]], list[float]]],
                             instrument: Callable[[list[float]], list[float]],
                             deviation_calculator: DeviationCalculatorInterface,
                             melee_range: float = float(16)) -> tuple[list[float], float]:

    return ballistic_optimize(taylor_coeff, 
                              get_random_melee_ballistic_device(len(taylor_coeff), melee_range), 
                              coeff_functionizer,
                              instrument,
                              deviation_calculator)

def rangemelee_ballistic_optimize(taylor_coeff: list[float],
                                  coeff_functionizer: Callable[[list[float]], Callable[[list[float]], list[float]]],
                                  instrument: Callable[[list[float]], list[float]],
                                  deviation_calculator: DeviationCalculatorInterface,
                                  magnetic_range: float = float(16), melee_range: float = float(16)):

    return ballistic_optimize(taylor_coeff, 
                              get_random_rangemelee_ballistic_device(len(taylor_coeff), magnetic_range, melee_range), 
                              coeff_functionizer,
                              instrument,
                              deviation_calculator)

def train(taylor_coeff: list[float],
          coeff_functionizer: Callable[[list[float]], Callable[[list[float]], list[float]]],
          instrument: Callable[[list[float]], list[float]], instrument_x_range: list[float], instrument_sampling_sz: int,
          directional_optimization_sz: int, training_epoch_sz: int) -> list[float]:

    optimizing_taylor_model: list[float]                = copy.deepcopy(taylor_coeff)
    deviation_calculator: DeviationCalculatorInterface  = RandomDiscreteDeviationCalculator(instrument_x_range, instrument_sampling_sz)

    for _ in range(training_epoch_sz):
        round_rs: list[tuple[list[float], float]] = []

        for __ in range(directional_optimization_sz):
            random_value: int = random.randrange(0, 3)

            try:
                if random_value == 0:
                    (new_taylor_model, deviation_hint) = range_ballistic_optimize(optimizing_taylor_model, coeff_functionizer, instrument, deviation_calculator, (random.random() * 10) ** random.randrange(0, 10))
                    round_rs += [(deviation_hint, new_taylor_model)]
                elif random_value == 1:
                    (new_taylor_model, deviation_hint) = melee_ballistic_optimize(optimizing_taylor_model, coeff_functionizer, instrument, deviation_calculator, (random.random() * 10) ** random.randrange(0, 10))
                    round_rs += [(deviation_hint, new_taylor_model)]
                elif random_value == 2:
                    (new_taylor_model, deviation_hint) = rangemelee_ballistic_optimize(optimizing_taylor_model, coeff_functionizer, instrument, deviation_calculator, (random.random() * 10) ** random.randrange(0, 10), (random.random() * 10) ** random.randrange(0, 10))
                    round_rs += [(deviation_hint, new_taylor_model)]

            except Exception as e:
                print(e)

        if len(round_rs) == 0:
            continue

        (best_deviation_hint, cand_taylor_model)    = min(round_rs)
        best_deviation_real                         = deviation_calculator.deviation(coeff_functionizer(cand_taylor_model), instrument)
        current_deviation                           = deviation_calculator.deviation(coeff_functionizer(optimizing_taylor_model), instrument)

        if best_deviation_real < current_deviation:
            optimizing_taylor_model = cand_taylor_model
            print("update", best_deviation_real)
        else:
            print("keep", current_deviation)

    return optimizing_taylor_model

def main():

    #I dont know if yall see what I see yet we are very hopeful about the future (for the very first time (1990s), we have solved the problem of stable diffusion + sound-alike effect from naive linear)

    #Taylor Series is actually the way
    #I was thinking of 1 var, 2 var, 4 var, 8 var
    #the sequence of those Taylor Series' length, the centrality add operation, rotate, rinse and repeat
    #we are gonna need a shii (I dont know why people pronounce it that way) ton of compute, a platform to run this on, yet I think its gonna be very compact, we are heading in the right direction

    #I was proving the difference between a multivariate projection vs 1 dimensional projections + add operation
    #here is the twist, add operation is another projection, alright I know this sounds silly, so is it all Taylor Series, yes, it's possible to create everything out of Taylor Series

    #proof by induction:

    #assume 1 dimensional projection + add suffice f(x) + f(x1) == f(x, x1)
    #assume we are to train new variable <x> such that <x, y> -> C and <x, y1> -> C1 for y and y1 are two points c <the_already_trained_set>
    #x now has to be of two values to keep the induction going

    #its the art of variable intercourse, we give off <DNA> information by doing multidimensional projections or add operation
    #we dont really know the exact formula for this operation, we just know the regex form of all centrality-differential-intellect-based, maybe we'll balance between reality (compute limits, resource, money vs benefits, overheads, etc.) and theoretical from there

    #recall when we noticed the problem of centrality, creating a new word, that word numerical range is not always within the computable model range, such creates anomaly + skewness + requires attention to solve the problem
    #we attempt to solve the problem by using water, we pour water over the projection space + take deviation to steer the course

    taylor_approx: TaylorApprox                         = make_taylor_model(3, 3, 3)
    taylor_coeff: list[list[float]]                     = taylor_approx.coeff_dump()
    flattened_coeff: list[float]                        = flatten(taylor_coeff)

    range_arr: list[float]                              = [float(16), float(16), float(16)]
    sampling_sz: int                                    = 64
    # deviation_calculator: DeviationCalculatorInterface  = RandomDiscreteDeviationCalculator(range_arr, sampling_sz)
    directional_optimization_sz: int                    = 128
    training_epoch_sz: int                              = 1 << 20

    def instrument(x: list[float]) -> list[float]:
        return [3 * x[0] * x[1], 2 * x[1] * x[2], x[2]]

    def _coeff_functionizer(coeff: list[float]) -> Callable[[list[float]], list[float]]:
        unflattened_coeff: list[list[float]]    = shape_as(coeff, get_no_jagged_shape(taylor_coeff))
        new_approxer: TaylorApprox              = taylor_approx.clone() 
        new_approxer.coeff_load(unflattened_coeff)

        return CallableTaylorApprox(new_approxer)     

    print(train(flattened_coeff, _coeff_functionizer, instrument, range_arr, sampling_sz, directional_optimization_sz, training_epoch_sz))

main()