import itertools
import random

import numpy as np


class Element:
    def __init__(self, diagonal_matrix_entries, permutation):
        self.permutation = permutation
        self.diagonal_matrix_entries = diagonal_matrix_entries
        self.composite_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for i in range(3):
            self.composite_matrix[i][permutation[i]] = diagonal_matrix_entries[permutation[i]]
        self.pairwise_notation = [diagonal_matrix_entries, permutation]
        self.composite_matrix_np = np.array(self.composite_matrix)
        self.S3labels = {'0': [0, 1, 2], '1': [1, 0, 2], '2': [0, 2, 1], '3': [2, 0, 1], '4': [1, 2, 0], '5': [2, 1, 0]}
        self.left_hand_label = f"({int(sum((self.diagonal_matrix_entries[i] - 1) / (-2) * 2 ** i for i in [0, 1, 2]))}," \
                               f"{list(self.S3labels.keys())[list(self.S3labels.values()).index(self.permutation)]})"
        self.right_hand_label = f"({int(sum((self.diagonal_matrix_entries[self.permutation[i]] - 1) / (-2) * 2 ** i for i in [0, 1, 2]))}," \
                                f"{list(self.S3labels.keys())[list(self.S3labels.values()).index(self.permutation)]})"

    def __mul__(self, other):
        return Element(
            [self.diagonal_matrix_entries[i] * other.diagonal_matrix_entries[self.permutation[i]] for i in range(3)],
            [other.permutation[self.permutation[i]] for i in range(3)])


class Group:
    def __init__(self, elements=None):
        self.elements = elements
        self.neutral = self.elements["(0,0)"]
        self.order = len(self.elements)
        self.inverseDict = {}
        for label, element in self.elements.items():
            inverse_label_list = list(self.inverseDict.keys())
            inverse_elements_list = list(self.inverseDict.items())
            if label in inverse_elements_list:
                inverse_index = inverse_elements_list.index(label)
                inverse_label = inverse_label_list[inverse_index]
                self.inverseDict.update({label: inverse_label})
            else:
                neutral = self.elements["(0,0)"]
                for potential_inverse_label, potential_inverse in self.elements.items():
                    product = element*potential_inverse
                    if product.composite_matrix == neutral.composite_matrix:
                        self.inverseDict.update({label: potential_inverse_label})
        self.orderDict = {}
        for label, element in self.elements.items():
            i = 1
            product = element
            while product.composite_matrix != self.neutral.composite_matrix:
                product *= element
                i += 1

            self.orderDict.update({label: i})
        self.subgroupDict = {}
        self.cosetDict = {}

    def addSubgroup(self, label, subgroup):
        if isinstance(subgroup, Group):
            for label_1 in subgroup.elements.keys():
                for label_2 in subgroup.elements.keys():
                    product = Full_Octahedral_Group.elements[label_1] * Full_Octahedral_Group.elements[label_2]
                    key_list = list(self.elements.keys())
                    elements_list = list(self.elements.values())
                    matrix_list = [element.composite_matrix for element in elements_list]
                    index = matrix_list.index(product.composite_matrix)
                    if key_list[index] not in subgroup.elements:
                        print(
                            f"Invalid subgroup {label}: {label_1} times {label_2} gives {key_list[index]}, {matrix_list[index]}")
                        return
            self.subgroupDict.update({label: subgroup})
            self.findCoset(subgroup)
        else:
            print("Check input typing")

    def findCoset(self, subgroup):
        if isinstance(subgroup, Group):
            subgroup_list = list(self.subgroupDict.values())
            index = subgroup_list.index(subgroup)
            label = list(self.subgroupDict.keys())[index]
            coset = []
            spent_elements = []
            group_order = self.order
            subgroup_order = subgroup.order
            coset_amounts = group_order // subgroup_order
            while len(coset) < coset_amounts:
                for trial_element in self.elements.values():
                    if trial_element.composite_matrix not in spent_elements:
                        for subgroup_element in subgroup.elements.values():
                            product = trial_element * subgroup_element
                            if subgroup_element == subgroup.neutral:
                                coset.append(trial_element)
                            spent_elements.append(product.composite_matrix)

            self.cosetDict.update({label: coset})


class Block:
    def __init__(self, base_point, body_diagonal_vector):
        self.base_point = base_point
        self.body_diagonal_vector = body_diagonal_vector
        self.ranges = []
        for n, x in enumerate(self.base_point):
            temp_list = [self.base_point[n] + self.body_diagonal_vector[n], self.base_point[n]]
            if temp_list[1] < temp_list[0]:
                temp_list.reverse()
            self.ranges.append(temp_list)
        self.set = self.makeSet()
        self.greatest_length = max(self.ranges[i][1] - self.ranges[i][0] for i in range(3))

    def __str__(self):
        return f"[{self.ranges[0]}] X [{self.ranges[1]}] X [{self.ranges[2]}]"

    def volume(self):
        Volume = 1
        for interval in self.ranges:
            Volume *= interval[1] - interval[0]
        return Volume

    def makeSet(self):
        temp_set = set(())

        x_range = self.ranges[0]
        y_range = self.ranges[1]
        z_range = self.ranges[2]

        for x in range(x_range[0], x_range[1]):
            for y in range(y_range[0], y_range[1]):
                for z in range(z_range[0], z_range[1]):
                    temp_set.add(f"{x + 0.5},{y + 0.5},{z + 0.5}")

        return temp_set

    def contains_vector(self, vector):
        return all(self.ranges[i][0] <= vector[i] <= self.ranges[i][1] for i in range(3))

    def reorient(self, orientation):
        new_ranges = [0, 0, 0]
        orientation_pair = orientation.pairwise_notation
        for n, interval in enumerate(self.ranges):
            new_index = orientation_pair[1][n]
            is_flipped = (orientation_pair[0][new_index] == -1)
            new_interval = interval
            if is_flipped is True:
                new_interval = [-x for x in new_interval]
                if new_interval[0] > new_interval[1]:
                    new_interval.reverse()
            new_ranges[new_index] = new_interval
            is_flipped = False
        minkowski_difference_base_point = []
        minkowski_difference_body_diagonal = []
        for interval in new_ranges:
            minkowski_difference_base_point.append(interval[0])
            minkowski_difference_body_diagonal.append(interval[1] - interval[0])
        return Block(minkowski_difference_base_point, minkowski_difference_body_diagonal)


class MultiBlock:
    def __init__(self, base_block=None, mark_block=None, group=None, subgroup=None):
        self.block_list = []
        self.base_block = Block(base_block[0], base_block[1])
        if mark_block is not None:
            self.mark_block = Block(mark_block[0], mark_block[1])
        else:
            self.mark_block = Block([0,0,0],[0,0,0])
        self.group = group
        self.subgroup = subgroup
        self.current_orientation = self.group.neutral
        self.orientable_block_list = []
        self.block_parameter_list = [base_block, mark_block]
        self.block_list.append(self.base_block)
        self.orientable_block_list.append(self.base_block)
        if mark_block is not None:
            for element in self.subgroup.elements.values():
                Block1 = Block(self.block_parameter_list[1][0], self.block_parameter_list[1][1])
                Block1 = Block1.reorient(element)
                Block_Boundaries = Block1.ranges
                Present_Boundaries = [block.ranges for block in self.block_list]
                if Block_Boundaries not in Present_Boundaries:
                    self.block_list.append(Block1)
                    self.orientable_block_list.append(
                        Block(self.block_parameter_list[1][0], self.block_parameter_list[1][1]).reorient(element))



        self.minkowski_difference_dictionary = {}
        self.DOMB_size = len(self.block_list)
        self.volume = sum(block.volume() for block in self.block_list)
        self.getMinkowskiDifferences()

    def return_orientation(self):
        current_orientation = self.current_orientation
        inverse_orientation = self.group.inverseDict[current_orientation.right_hand_label]
        return self.group.elements[inverse_orientation]

    def reorient(self, orientation=None):
        if orientation is None:
            orientation = self.group.neutral
        for n, block in enumerate(self.block_list):
            self.orientable_block_list[n] = block.reorient(orientation)
        self.current_orientation = orientation

    def __str__(self):
        print_string = ""
        Amount_of_blocks = len(self.block_list)
        for block in self.block_list:
            print_string += str(block)
            if Amount_of_blocks > 1:
                print_string += ", "
                Amount_of_blocks -= 1
        return print_string

    def intersect(self, block_1_index, block_2_index):
        intersection = []
        for axis_number in [0, 1, 2]:
            interval_1 = self.block_list[block_1_index].ranges[axis_number]
            interval_2 = self.block_list[block_2_index].ranges[axis_number]
            if (interval_1[1] < interval_2[0]) or interval_2[1] < interval_1[0]:
                intersection.append([])
            if interval_2[0] < interval_1[1]:
                if interval_1[1] < interval_2[1]:
                    intersection.append([interval_2[0], interval_1[1]])
                else:
                    intersection.append([interval_2[0], interval_2[1]])
            else:
                if interval_1[1] < interval_2[1]:
                    intersection.append([interval_1[0], interval_1[1]])
                else:
                    intersection.append([interval_1[0], interval_2[1]])

        intersection_base_point = []
        intersection_body_diagonal = []
        for interval in intersection:
            intersection_base_point.append(interval[0])
            intersection_body_diagonal.append(interval[1] - interval[0])
        return Block(intersection_base_point, intersection_body_diagonal)

    def minkowski_difference(self, block_1, block_2):
        minkowski_difference_set = []
        for axis_number in [0, 1, 2]:
            interval_1 = block_1.ranges[axis_number]
            interval_2 = block_2.ranges[axis_number]
            interval = [interval_1[0] - interval_2[1], interval_1[1] - interval_2[0]]
            if interval[1] < interval[0]:
                interval.reverse()
            minkowski_difference_set.append(interval)

        minkowski_difference_base_point = []
        minkowski_difference_body_diagonal = []
        for interval in minkowski_difference_set:
            minkowski_difference_base_point.append(interval[0])
            minkowski_difference_body_diagonal.append(interval[1] - interval[0])
        return Block(minkowski_difference_base_point, minkowski_difference_body_diagonal)

    def getMinkowskiDifferences(self, orientation=None):
        if orientation is None:
            orientation = self.group.neutral
        self.reorient(orientation)

        for block_index_1 in range(self.DOMB_size):
            for block_index_2 in range(self.DOMB_size):
                self.minkowski_difference_dictionary.update({f"({block_index_1},{block_index_2})":
                    self.minkowski_difference(
                        self.block_list[block_index_1]
                        , self.orientable_block_list[block_index_2])})

    def getExcludedVolume(self, orientation=None):
        if orientation is None:
            orientation = self.group.neutral
        self.getMinkowskiDifferences(orientation)
        Total_Minkowski_difference = set(())
        for Minkowski_Difference in self.minkowski_difference_dictionary.values():
            for volumeElement in Minkowski_Difference.set:
                Total_Minkowski_difference.add(volumeElement)
        return len(Total_Minkowski_difference)

    def MonteCarloExcludedVolumes(self, n=10**6):
        Length = 2*(self.base_block.greatest_length + self.mark_block.greatest_length)
        print(Length)
        k = 0
        for i in range(n):
            Translation = [random.uniform(-Length, Length), random.uniform(-Length, Length), random.uniform(-Length, Length)]
            Test = any(block.contains_vector(Translation) for block in self.minkowski_difference_dictionary.values())
            if Test == True:
                k += 1
        return k/n*((2*Length)**3)
    def calculateExcludedVolumes(self):
        index = list(self.group.subgroupDict.values()).index(self.subgroup)
        label = list(self.group.subgroupDict.keys())[index]
        cosets = self.group.cosetDict[label]
        excluded_volumes = {}
        for coset in cosets:
            label = list(self.group.elements.keys())[list(self.group.elements.values()).index(coset)]
            excluded_volumes.update({label: self.getExcludedVolume(coset)})
        return excluded_volumes

    def verifyGroupExclusivity(self):
        self.reorient(self.group.neutral)
        Block_Ranges = [block.ranges for block in self.block_list]
        Base = Block_Ranges.pop(0)
        i = 0
        n = self.group.order
        for element in self.group.elements.values():
            self.reorient(self.group.neutral)
            self.reorient(element)
            Reoriented_Ranges = [block.ranges for block in self.orientable_block_list]
            Reoriented_base = Reoriented_Ranges.pop(0)
            Unoriented_Base = self.base_block

            check1 = Unoriented_Base.ranges == Reoriented_base
            check2 = all(block in Reoriented_Ranges for block in Block_Ranges)
            if element in self.subgroup.elements.values():
                if (check1 and check2) is True:
                    #print("Valid MultiBlock")
                    i += 1
                else:
                    print(f"Invalid for {element.right_hand_label},\n{[Block_Ranges[0], Reoriented_Ranges]}")
            else:
                if (check1 and check2) is False:
                    #print("Valid Multiblock")
                    i += 1
                else:
                    print(f"Invalid for {element.right_hand_label},\n{[Block_Ranges, Reoriented_Ranges]}")

        if i == n:
            print("Validation complete: Multiblock is valid.")

PlusMinusOne = [1, -1]

Diagonal_Matrix_Entry_Possibilities = [[entry_1, entry_2, entry_3] for entry_1 in PlusMinusOne for entry_2 in
                                       PlusMinusOne for entry_3 in PlusMinusOne]

Permutations = [[0, 1, 2], [1, 0, 2], [0, 2, 1], [2, 0, 1], [1, 2, 0], [2, 1, 0]]
Full_Orthogonal_Group_Elements = {f"({int(sum((Diagonal_Matrix_Entry[Permutation[i]] - 1) / (-2) * 2 ** i for i in [0, 1, 2]))},{Permutation_index})": Element(Diagonal_Matrix_Entry, Permutation)
                                  for Diagonal_index, Diagonal_Matrix_Entry in
                                  enumerate(Diagonal_Matrix_Entry_Possibilities)
                                  for Permutation_index, Permutation in enumerate(Permutations)}

Full_Octahedral_Group = Group(Full_Orthogonal_Group_Elements)
Subgroup_Label_Dict = {"Oh": ["(0,0)", "(0,1)", "(0,2)", "(0,3)", "(0,4)", "(0,5)",
                              "(1,0)", "(1,1)", "(1,2)", "(1,3)", "(1,4)", "(1,5)",
                              "(2,0)", "(2,1)", "(2,2)", "(2,3)", "(2,4)", "(2,5)",
                              "(3,0)", "(3,1)", "(3,2)", "(3,3)", "(3,4)", "(3,5)",
                              "(4,0)", "(4,1)", "(4,2)", "(4,3)", "(4,4)", "(4,5)",
                              "(5,0)", "(5,1)", "(5,2)", "(5,3)", "(5,4)", "(5,5)",
                              "(6,0)", "(6,1)", "(6,2)", "(6,3)", "(6,4)", "(6,5)",
                              "(7,0)", "(7,1)", "(7,2)", "(7,3)", "(7,4)", "(7,5)"],
                       "O": ["(0,0)", "(0,3)", "(0,4)",
                             "(1,1)", "(1,2)", "(1,5)",
                             "(2,1)", "(2,2)", "(2,5)",
                             "(3,0)", "(3,3)", "(3,4)",
                             "(4,1)", "(4,2)", "(4,5)",
                             "(5,0)", "(5,3)", "(5,4)",
                             "(6,0)", "(6,3)", "(6,4)",
                             "(7,1)", "(7,2)", "(7,5)"],
                       "S4": ["(0,0)", "(0,1)", "(0,2)", "(0,3)", "(0,4)", "(0,5)",
                              "(3,0)", "(3,1)", "(3,2)", "(3,3)", "(3,4)", "(3,5)",
                              "(5,0)", "(5,1)", "(5,2)", "(5,3)", "(5,4)", "(5,5)",
                              "(6,0)", "(6,1)", "(6,2)", "(6,3)", "(6,4)", "(6,5)"],
                       "A4C2": ["(0,0)", "(0,3)", "(0,4)",
                                "(1,0)", "(1,3)", "(1,4)",
                                "(2,0)", "(2,3)", "(2,4)",
                                "(3,0)", "(3,3)", "(3,4)",
                                "(4,0)", "(4,3)", "(4,4)",
                                "(5,0)", "(5,3)", "(5,4)",
                                "(6,0)", "(6,3)", "(6,4)",
                                "(7,0)", "(7,3)", "(7,4)"],
                       "D4C2": ["(0,0)", "(0,1)",
                                "(1,0)", "(1,1)",
                                "(2,0)", "(2,1)",
                                "(3,0)", "(3,1)",
                                "(4,0)", "(4,1)",
                                "(5,0)", "(5,1)",
                                "(6,0)", "(6,1)",
                                "(7,0)", "(7,1)"],
                       "A4": ["(0,0)", "(0,3)", "(0,4)",
                              "(3,0)", "(3,3)", "(3,4)",
                              "(5,0)", "(5,3)", "(5,4)",
                              "(6,0)", "(6,3)", "(6,4)"],
                       "D6": ["(0,0)", "(0,1)", "(0,2)", "(0,3)", "(0,4)", "(0,5)",
                              "(7,0)", "(7,1)", "(7,2)", "(7,3)", "(7,4)", "(7,5)"],
                       "C23_1": ["(0,0)", "(1,0)", "(2,0)", "(3,0)",
                                 "(4,0)", "(5,0)", "(6,0)", "(7,0)"],
                       "C23_2": ["(0,0)", "(0,1)",
                                 "(3,0)", "(3,1)",
                                 "(4,0)", "(4,1)",
                                 "(7,0)", "(7,1)"],
                       "C4C2": ["(0,0)",
                                "(1,1)",
                                "(2,1)",
                                "(3,0)",
                                "(4,0)",
                                "(5,1)",
                                "(6,1)",
                                "(7,0)"],
                       "D4_1": ["(0,0)",
                                "(1,1)",
                                "(2,1)",
                                "(3,0)",
                                "(4,1)",
                                "(5,0)",
                                "(6,0)",
                                "(7,1)"],
                       "D4_2": ["(0,0)", "(0,1)",
                                "(1,0)", "(1,1)",
                                "(2,0)", "(2,1)",
                                "(3,0)", "(3,1)"],
                       "D4_3": ["(0,0)",
                                "(1,0)",
                                "(2,0)",
                                "(3,0)",
                                "(4,1)",
                                "(5,1)",
                                "(6,1)",
                                "(7,1)"],
                       "D4_4": ["(0,0)", "(0,1)",
                                "(3,0)", "(3,1)",
                                "(5,0)", "(5,1)",
                                "(6,0)", "(6,1)"],

                       "D3_1": ["(0,0)", "(7,1)", "(7,2)", "(0,3)", "(0,4)", "(7,5)"],
                       "D3_2": ["(0,0)", "(0,1)", "(0,2)", "(0,3)", "(0,4)", "(0,5)"],

                       "C6": ["(0,0)",  "(0,3)", "(0,4)", "(7,0)",  "(7,3)", "(7,4)"],
                       "C4_1": ["(0,0)",
                                "(1,1)",
                                "(2,1)",
                                "(3,0)"],
                       "C4_2": ["(0,0)",
                                "(3,0)",
                                "(5,1)",
                                "(6,1)"],
                       "K4_1": ["(0,0)", "(3,0)",
                                "(5,0)", "(6,0)"],

                       "K4_2": ["(0,0)", "(3,0)",
                                "(4,1)", "(7,1)"],
                       "K4_3": ["(0,0)", "(1,0)",
                                "(2,0)", "(3,0)"],
                       "K4_4": ["(0,0)", "(0,1)",
                                "(3,0)", "(3,1)"],
                       "K4_5": ["(0,0)", "(0,1)",
                                "(4,0)", "(4,1)"],

                       "K4_6": ["(0,0)",
                                "(3,0)",
                                "(4,0)",
                                "(7,0)"],
                       "K4_7": ["(0,0)", "(0,1)",
                                "(7,0)", "(7,1)"],

                       "C3": ["(0,0)", "(0,3)", "(0,4)"],
                       "C2_1": ["(0,0)", "(3,0)"],
                       "C2_2": ["(0,0)", "(7,1)"],
                       "C2_3": ["(0,0)", "(4,0)"],
                       "C2_4": ["(0,0)", "(0,1)"],
                       "C2_5": ["(0,0)", "(7,0)"],

                       "C1": ["(0,0)"]


                       }
Subgroup_Element_Dicts = {label: {elem_label: Full_Orthogonal_Group_Elements[elem_label] for elem_label in elem_labels} for label, elem_labels in Subgroup_Label_Dict.items()}
Subgroup_Dict = {label: Group(elements) for label, elements in Subgroup_Element_Dicts.items()}

for label, subgroup in Subgroup_Dict.items():
    Full_Octahedral_Group.addSubgroup(label, subgroup)


#Parameter settings
a = 2
b = 3
c = 4
d = 1
e = 2
f = 3
Subgroup = "O"

Full_Octahedral_Group.findCoset(Full_Octahedral_Group.subgroupDict[Subgroup])

Base_Block_Dict = {"Bc": [[-a, -a, -a], [2*a, 2*a, 2*a]], "Bs": [[-a, -a, -c], [2*a, 2*a, 2*c]], "Br": [[-a, -b, -c], [2*a, 2*b, 2*c]]}
Marker_Block_Dict = {"Bcs": [[-d,-d,c], [2*d, 2*d, 2*f]], "Bcr": [[-d,-e,c], [2*d, 2*e, 2*f]], 'Best': [[-d,b,c], [2*d, -2*d, 2*f]],
                     'Bert': [[-d,b,c], [2*d, -2*e, 2*f]], 'Bess': [[-d,b,c], [2*d, 2*d, -2*f]], 'Bers': [[-d,b,c], [2*d, 2*e, -2*f]],
                     'Bvst': [[a,b,c], [-2*d, -2*d, 2*f]], 'Bvrt': [[a,b,c], [-2*d, -2*e, 2*f]], 'Bvss': [[a,b,c], [-2*d, 2*d, -2*f]],
                     'Bvrs': [[a,b,c], [-2*d, 2*e, -2*f]]
                     }

Multiblock_Dict = {"Oh": MultiBlock(Base_Block_Dict["Bc"], None, Full_Octahedral_Group, Subgroup_Dict['Oh']),
                   "O":  MultiBlock(Base_Block_Dict["Bc"], Marker_Block_Dict['Bvrt'], Full_Octahedral_Group, Subgroup_Dict['O']),
                   "S4":  MultiBlock(Base_Block_Dict["Bc"], Marker_Block_Dict['Bvst'], Full_Octahedral_Group, Subgroup_Dict['S4']),
                   "A4C2": MultiBlock(Base_Block_Dict["Bc"], Marker_Block_Dict['Bcr'], Full_Octahedral_Group, Subgroup_Dict['A4C2']),
                   "D4C2": MultiBlock(Base_Block_Dict["Bs"], None, Full_Octahedral_Group,
                                      Subgroup_Dict['D4C2']),
                   "A4": MultiBlock(Base_Block_Dict["Bc"], Marker_Block_Dict['Bvrt'], Full_Octahedral_Group,
                                      Subgroup_Dict['A4']),
                   "D6": MultiBlock(Base_Block_Dict["Bc"], Marker_Block_Dict['Bvst'], Full_Octahedral_Group,
                                      Subgroup_Dict['D6']),
                   "C23_1": MultiBlock(Base_Block_Dict["Br"], None, Full_Octahedral_Group,
                                      Subgroup_Dict['C23_1']),
                   "C23_2": MultiBlock(Base_Block_Dict["Bs"], Marker_Block_Dict['Bvrs'], Full_Octahedral_Group,
                                      Subgroup_Dict['C23_2']),
                   "C4C2": MultiBlock(Base_Block_Dict["Bs"], Marker_Block_Dict['Bvrs'], Full_Octahedral_Group,
                                      Subgroup_Dict['C4C2']),
                   "D4_1": MultiBlock(Base_Block_Dict["Bs"], Marker_Block_Dict['Bvrs'], Full_Octahedral_Group,
                                      Subgroup_Dict['D4_1']),
                   "D4_2": MultiBlock(Base_Block_Dict["Bs"], Marker_Block_Dict['Bcs'], Full_Octahedral_Group,
                                      Subgroup_Dict['D4_2']),
                   "D4_3": MultiBlock(Base_Block_Dict["Bs"], Marker_Block_Dict['Bert'], Full_Octahedral_Group,
                                      Subgroup_Dict['D4_3']),
                   "D4_4": MultiBlock(Base_Block_Dict["Bs"], Marker_Block_Dict['Bvst'], Full_Octahedral_Group,
                                      Subgroup_Dict['D4_4']),
                   "C6": MultiBlock(Base_Block_Dict["Bc"], Marker_Block_Dict['Bvrt'], Full_Octahedral_Group,
                                      Subgroup_Dict['C6']),
                   "D3_1": MultiBlock(Base_Block_Dict["Bc"], Marker_Block_Dict['Bvrt'], Full_Octahedral_Group,
                                      Subgroup_Dict['D3_1']),
                   "D3_2": MultiBlock(Base_Block_Dict["Bc"], Marker_Block_Dict['Bvst'], Full_Octahedral_Group,
                                      Subgroup_Dict['D3_2']),
                   "C4_1": MultiBlock(Base_Block_Dict["Bs"], Marker_Block_Dict['Bvrt'], Full_Octahedral_Group,
                                      Subgroup_Dict['C4_1']),
                   "C4_2": MultiBlock(Base_Block_Dict["Bs"], Marker_Block_Dict['Bvrt'], Full_Octahedral_Group,
                                      Subgroup_Dict['C4_2']),
                   "K4_1": MultiBlock(Base_Block_Dict["Br"], Marker_Block_Dict['Bvrt'], Full_Octahedral_Group,
                                      Subgroup_Dict['K4_1']),
                   "K4_2": MultiBlock(Base_Block_Dict["Bs"], Marker_Block_Dict['Bvrs'], Full_Octahedral_Group,
                                      Subgroup_Dict['K4_2']),
                   "K4_3": MultiBlock(Base_Block_Dict["Br"], Marker_Block_Dict['Bcr'], Full_Octahedral_Group,
                                      Subgroup_Dict['K4_3']),
                   "K4_4": MultiBlock(Base_Block_Dict["Bs"], Marker_Block_Dict['Bvrs'], Full_Octahedral_Group,
                                      Subgroup_Dict['K4_4']),
                   "K4_5": MultiBlock(Base_Block_Dict["Bs"], Marker_Block_Dict['Bvst'], Full_Octahedral_Group,
                                      Subgroup_Dict['K4_5']),
                   "K4_6": MultiBlock(Base_Block_Dict["Br"], Marker_Block_Dict['Bert'], Full_Octahedral_Group,
                                      Subgroup_Dict['K4_6']),
                   "K4_7": MultiBlock(Base_Block_Dict["Bs"], Marker_Block_Dict['Bvrs'], Full_Octahedral_Group,
                                      Subgroup_Dict['K4_7']),
                   "C3": MultiBlock(Base_Block_Dict["Bc"], Marker_Block_Dict['Bvrt'], Full_Octahedral_Group,
                                      Subgroup_Dict['C3']),
                   "C2_1": MultiBlock(Base_Block_Dict["Br"], Marker_Block_Dict['Bvrt'], Full_Octahedral_Group,
                                      Subgroup_Dict['A4C2']),
                   "C2_2": MultiBlock(Base_Block_Dict["Bs"], Marker_Block_Dict['Bvrs'], Full_Octahedral_Group,
                                      Subgroup_Dict['C2_2']),
                   "C2_3": MultiBlock(Base_Block_Dict["Br"], Marker_Block_Dict['Bcr'], Full_Octahedral_Group,
                                      Subgroup_Dict['C2_3']),
                   "C2_4": MultiBlock(Base_Block_Dict["Bs"], Marker_Block_Dict['Bvrs'], Full_Octahedral_Group,
                                      Subgroup_Dict['C2_4']),
                   "C2_5": MultiBlock(Base_Block_Dict["Br"], Marker_Block_Dict['Bvrt'], Full_Octahedral_Group,
                                      Subgroup_Dict['C2_5']),
                   "C1": MultiBlock(Base_Block_Dict["Br"], Marker_Block_Dict['Bvrt'], Full_Octahedral_Group,
                                      Subgroup_Dict['C1'])

                   }

Base_block = [[-a, -b, -c], [2*a, 2*b, 2*c]]
Marker_block = [[-d,-e,c], [2*d, 2*e, 2*f]]
D = Multiblock_Dict[Subgroup]
D.verifyGroupExclusivity()
for element in D.group.cosetDict[Subgroup]:
    print(f"Coset {element.left_hand_label}")
    for element2 in D.group.subgroupDict[Subgroup].elements.values():
        element3 = element*element2
        print(element3.left_hand_label)
D.getMinkowskiDifferences(D.group.elements["(4,1)"])
for Difference in D.minkowski_difference_dictionary.keys():
    print(Difference, "-->", D.minkowski_difference_dictionary[Difference])
D.reorient(D.group.elements["(0,0)"])

D.reorient(D.group.elements["(0,0)"])
EV = D.calculateExcludedVolumes()
# for Label, Volume in EV.items():
#     print(f"{Label}: {Volume}")
D.reorient(D.group.elements["(0,0)"])
cosets = list(D.group.cosetDict[Subgroup])
D.reorient(D.group.elements["(0,0)"])
# for c_index in range(len(cosets)):
#     D.reorient(D.group.neutral)
#     D.getMinkowskiDifferences(cosets[c_index])
#     #MCvolume = D.MonteCarloExcludedVolumes(n=10**6)
#     Tev = D.getExcludedVolume(cosets[c_index])
#     print(f'{cosets[c_index].right_hand_label}, {Tev}')
#     #print(MCvolume)
#     #print((MCvolume - Tev)/Tev*100, "%")

print(D.group.elements["(4,5)"].left_hand_label)