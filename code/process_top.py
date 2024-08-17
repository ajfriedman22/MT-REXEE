import numpy as np
import pandas as pd
import os
import re
import argparse
import json

def swap_name(init_atom_names, new_resname, df_top):
    """
    Determine the corresponding atom name in new molecule

    Parameters
    ----------
    init_atom_names : list of str
        Atom name in the original molecule
    new_resname : str
        Resname for the new molecule
    df_top : pd.DataFrame
        Dataframe containing the connectivity between the atoms in each molecule

    Return
    ------
    new_atom_names : list of str
        Atom names in the new molecule
    """
    #Find all atom names in new moleucle
    new_names = set(df_top[df_top['Resname'] == new_resname]['Connect 1'].to_list() + df_top[df_top['Resname'] == new_resname]['Connect 1'].to_list() + df_top[df_top['Resname'] == new_resname]['Connect 2'].to_list() + df_top[df_top['Resname'] == new_resname]['Connect 2'].to_list()) 
    new_atom_names = []
    for atom in init_atom_names:
        if atom in new_names:
            new_atom_names.append(atom)
            continue
        atom_num = re.findall(r'[0-9]+', atom)[0]
        atom_identifier = re.findall(r'[a-zA-Z]+', atom)[0]
        if 'D' in atom_identifier:
            atom_identifier = atom_identifier.strip('D')
        if 'V' in atom_identifier:
            atom_identifier = atom_identifier.strip('V')
        if f'{atom_identifier}V{atom_num}' in new_names:
            new_atom_names.append(f'{atom_identifier}V{atom_num}')
        elif f'{atom_identifier}{atom_num}' in new_names:
            new_atom_names.append(f'{atom_identifier}{atom_num}')
        elif f'D{atom_identifier}{atom_num}' in new_names:
            new_atom_names.append(f'D{atom_identifier}{atom_num}')
        else:
            raise Exception(f'Compatible atom could not be found for {atom}')
    return new_atom_names

def get_names(input):
    """
    Determine the names of all atoms in the topology and which lambda state for which they are dummy atoms

    Parameters
    ----------
    input : List of str
        Read the lines of the input topology

    Return
    ------
    start_line : int
        The next line to start reading from the topology
    atom_name : list of str
        All atom names in the topology
    state : list of int
        The state that the atom is a dummy atom (lambda=0, lambda=1, or -1 if nevver dummy)
    """
    atom_section = False
    atom_name, state = [], []
    for l, line in enumerate(input):
        if atom_section:
            line_sep = line.split(' ')
            if line_sep[0] == ';':
                continue
            while '' in line_sep:
                line_sep.remove('')
            if line_sep[0] == '\n':
                start_line = l+2
                break
            atom_name.append(line_sep[4])
            if len(line_sep) < 10:
                state.append(-1)
            elif float(line_sep[6]) == 0:
                state.append(0)
            elif float(line_sep[9]) == 0:
                state.append(1)
            else:
                state.append(-1)
        if line == '[ atoms ]\n':
            atom_section = True
    return start_line, atom_name, state

def deter_connection(main_only, other_only, main_name, other_name, df_top, main_state):
    """
    Determine the connectivity of the missing atoms in the topology

    Parameters
    ----------
    main_only : list of str
        All atoms which can be found only in the molecule of interest
    other_only : list of str
        All atoms which can be found only in the other molecule
    main_name : str
        resname for the molecule of interest
    other_name : str
        resname for the other molecule
    df_top : pd.DataFrame
        Connectivity of the atoms in each molecule
    main_state : list of int
        Which lambda state are each atom in the molecule of interest in the dummy state
    Return
    ------
    df : pd.DataFrame
        Dataframes contains the missing atoms, the real anchor atom that connects them, and the atom to be used to determine the angle to place the missing atoms
    """
    miss, D2R, R2D = [],[],[]
    align_atom, angle_atom = [], []
    for atom in main_only:
        element = atom.strip('0123456789')
        real_element = element.strip('DV')
        num = atom.strip('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            
        if f'D{atom}' in other_only or f'{element}V{num}' in other_only:
            D2R.append(atom)
        elif f'{real_element}{num}' in other_only:
            R2D.append(atom)
        else:
            miss.append(atom)
    df_select = df_top[df_top['Resname']==main_name]
    #Seperate into each seperate functional group to be added
    anchor_atoms  = []
    
    for m_atom in miss:
        #Find what atoms the missing atom connects to
        connected_atoms = []
        for a in df_select[df_select['Connect 1'] == m_atom]['Connect 2'].values:
            connected_atoms.append(a)
        for a in df_select[df_select['Connect 2'] == m_atom]['Connect 1'].values:
            connected_atoms.append(a)
        
        #If the atom connects to non-missing atoms than keep these as anchors
        for a in connected_atoms:
            if a not in miss:
                anchor_atoms.append(a)

    #Seperate missing atoms connected to each anchor
    miss_sep, align_atoms, angle_atoms = [],[],[]
    for anchor in anchor_atoms:
        miss_anchor = []

        #Which missing atoms are connected to the anchor
        search = True
        included_atoms = [anchor]
        while search == True:
            found_atoms = list(df_select[(df_select['Connect 1'].isin(included_atoms)) & (df_select['Connect 2'].isin(miss))]['Connect 2'].values) + list(df_select[(df_select['Connect 2'].isin(included_atoms)) & (df_select['Connect 1'].isin(miss))]['Connect 1'].values)
            if len(found_atoms) == 0:
                search = False
            else:
                for atom in found_atoms:
                    miss_anchor.append(atom)
                    included_atoms.append(atom)
                    miss.remove(atom)
        included_atoms.remove(anchor)
        miss_sep.append(included_atoms)
        
        #Find atoms connected to the anchor which are real in main state, but dummy when the atoms we are building are real
        align_atom = list(df_select[(df_select['Connect 1'] == anchor) & (df_select['State 2'] != main_state) & (df_select['State 2'] != -1)]['Connect 2'].values) + list(df_select[(df_select['Connect 2'] == anchor) & (df_select['State 1'] != main_state) & (df_select['State 1'] != -1)]['Connect 1'].values)
        align_atoms.append(align_atom[0])

        #Find the atom to use for matching the angle to ensure that the dummy atom is added in the correct orientation
        ignore_atoms = anchor_atoms + included_atoms + align_atom
        angle_atom = list(df_select[(df_select['Connect 1'] == anchor) & (~df_select['Connect 2'].isin(ignore_atoms))]['Connect 2'].values) + list(df_select[(df_select['Connect 2'] == anchor) & (~df_select['Connect 1'].isin(ignore_atoms))]['Connect 1'].values)
        angle_atoms.append(angle_atom[-1])
    
    #Now let's figure out what these atoms are called in the other molecule
    anchor_atoms_B = swap_name(anchor_atoms, other_name, df_top)
    angle_atoms_B = swap_name(angle_atoms, other_name, df_top)
    align_atoms_B = swap_name(align_atoms, other_name, df_top)
    df = pd.DataFrame({'Swap A': main_name, 'Swap B': other_name, 'Connecting Atom Name A': anchor_atoms, 'Anchor Atom Name B': anchor_atoms_B, 'Alignment Atom A': align_atoms, 'Alignment Atom B': align_atoms_B, 'Angle Atom A': angle_atoms, 'Angle Atom B': angle_atoms_B})
    df['Missing Atom Name'] = np.NaN
    df['Missing Atom Name'] = df['Missing Atom Name'].astype(object)
    df['Missing Atom Name'] = miss_sep

    return df

def read_top(file_name, resname):
    """
    Read the topology to find the file containing the molecule of interest

    Parameters
    ----------
    file_name : str
        Name for the topology file
    resname : str
        Name of the residue of interest we are searching for

    Return
    ------
    input_file : list of str
        Contents of the topology file which contains the residue of interest
    """
    input_file = open(file_name).readlines()
    itp_files = []
    atom_sect = True
    for line in input_file:
        if line == '[ atoms ]\n':
            atom_sect = True
        if atom_sect:
            if line == '\n':
                atom_sect = False
            line_sep = line.split(' ')
            while '' in line_sep:
                line_sep.remove('') 
            if len(line_sep) > 4 and line_sep[3] == resname:
                return input_file
        if '#include' in line:
            line_sep = line.split(' ')
            while '' in line_sep:
                line_sep.remove('') 
            itp_files.append(line_sep[-1].strip('\n""'))
    for file in itp_files:
        if os.path.exists(file):
            input_file = open(file).readlines()
            atom_sect = False
            for line in input_file:
                if line == '[ atoms ]\n':
                    atom_sect = True
                if atom_sect:
                    if line == '\n':
                        break
                    line_sep = line.split(' ')
                    while '' in line_sep:
                        line_sep.remove('') 
                    if len(line_sep) > 4 and line_sep[3] == resname:
                        return input_file
    raise Exception(f'Residue {resname} can not be found in {file_name}')

#Load input options
parser = argparse.ArgumentParser(prog='Process_top',description='Takes a series of topology files and outputs the connectivity in the files and a dataframe map to guide conformational swaps between pairs of molecules')
parser.add_argument('-top', '-t', required=True, nargs='+', type=str, help='All topology files for molecule to process')
parser.add_argument('-resname', '-r', required=True, nargs='+', type=str, help='All residue names for the residue of interest in the molecules being processed')

args = parser.parse_args()

top_files = args.top
swap_res = args.resname
rep_swap_pattern = {0:1,1:0}, {1:1,2:0}, {2:1,3:0}, {3:1,4:0}

if not os.path.exists('residue_connect.csv'):
    df_top = pd.DataFrame()
    for f, file_name in enumerate(top_files):
        #Read file
        input = read_top(file_name, swap_res[f])

        #Determine the atom names corresponding to the atom numbers
        start_line, atom_name, state = get_names(input)
    
        #Determine the connectivity of all atoms
        connect_1, connect_2, state_1, state_2 = [], [], [], [] #Atom 1 and atom 2 which are connected and which state they are dummy atoms
        for l, line in enumerate(input[start_line:]):
            line_sep = line.split(' ')
            if line_sep[0] == ';':
                continue
            if line_sep[0] == '\n':
                break
            while '' in line_sep:
                line_sep.remove('')
            connect_1.append(atom_name[int(line_sep[0])-1])
            connect_2.append(atom_name[int(line_sep[1])-1])
            state_1.append(state[int(line_sep[0])-1])
            state_2.append(state[int(line_sep[1])-1])
        df = pd.DataFrame({'Resname': swap_res[f], 'Connect 1': connect_1, 'Connect 2': connect_2, 'State 1': state_1, 'State 2': state_2})
        df_top = pd.concat([df_top, df])
    df_top.to_csv('residue_connect.csv')
else:
    df_top = pd.read_csv('residue_connect.csv')
    print('skip')

if not os.path.exists('residue_swap_map.csv'):
    df_map = pd.DataFrame()

    for swap in rep_swap_pattern:
        #Determine atoms not present in both molecules
        X, Y = swap.keys()
        for A, B in zip([X, Y], [Y, X]):
            input_A = read_top(top_files[A], swap_res[A])
            start_line, A_name, state = get_names(input_A)
            input_B = read_top(top_files[B], swap_res[B])
            start_line, B_name, state = get_names(input_B)
        
            A_only = [x for x in A_name if x not in B_name]
            B_only = [x for x in B_name if x not in A_name]
        
            #Seperate real to dummy switches
            df = deter_connection(A_only, B_only, swap_res[A], swap_res[B], df_top, swap[A])
    
            df_map = pd.concat([df_map, df])
        
    df_map.to_csv('residue_swap_map.csv')



