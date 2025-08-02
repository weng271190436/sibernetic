import pprint
import os

SIMULATION_BOX = "simulation box"
POSITION = "position"
VELOCITY = "velocity"
CONNECTION = "connection"
MEMBRANES = "membranes"
PARTICLE_MEM_INDEX = "particleMemIndex"


class Configuration:
    """
    Class to hold configuration data
    """

    def __init__(self):
        self.simulation_box = []
        self.particles = {}
        self.particles = {}
        self.connections = {}
        self.membranes = {}
        self.particle_mem_index = {}

    def is_elastic(self, particle_id):
        """
        Check if a particle is elastic based on its type.
        """
        if particle_id in self.particles:
            particle_type = self.particles[particle_id].get(POSITION)[3]
            return int(particle_type) == 2 or int(particle_type) == 3
        return False

    def __str__(self):

        pos_count = 0
        vel_count = 0
        for i in self.particles:
            if POSITION in self.particles[i]:
                pos_count += 1
            if VELOCITY in self.particles[i]:
                vel_count += 1
            # print(f"Particle {i}: {self.particles[i]}: Positions: {pos_count}, Velocities: {vel_count}")

                
        velocity_count = len(self.particles.get(VELOCITY, []))
        return (
            f"Configuration with {len(self.particles)} particles (pos: {pos_count}, vel: {vel_count}), {len(self.connections)} connections, "
            f"{len(self.membranes)} membranes, and {len(self.particle_mem_index)} particle membrane indices."
        )

    def __repr__(self):
        return self.__str__()


def load_configuration_file(filename, verbose=False):
    """
    Load a configuration file
    """
    configuration = Configuration()
    current_section = None

    print("Loading configuration from:", os.path.abspath(filename))
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Configuration file {filename} does not exist.")

    with open(filename, "r") as file:
        pos_count = 0
        vel_count = 0

        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "[" in line:
                current_section = line.strip("[]")

            elif current_section == SIMULATION_BOX:
                configuration.simulation_box.append(float(line))
            elif current_section == POSITION:
                if verbose:
                    print("Checking position line: %s (%i)"% (line, pos_count))
                x, y, z, particle_type = line.split()
                configuration.particles[pos_count] = {POSITION: (float(x), float(y), float(z), float(particle_type))}
                pos_count += 1
            elif current_section == VELOCITY:
                if verbose:
                    print("Checking velocity line: %s (%i)"% (line, vel_count))
                vx, vy, vz, particle_type = line.split() 
                configuration.particles[vel_count][VELOCITY] = (float(vx), float(vy), float(vz), float(particle_type))
                if verbose:
                    print(f' Particle {vel_count} - pos: {configuration.particles[vel_count][POSITION]}; vel: {configuration.particles[vel_count][VELOCITY]}')
                # assert(configuration.particles[vel_count][POSITION][3] == float(particle_type))

                vel_count += 1
            else:
                pass

    pprint.pprint(configuration)

    return configuration


def write_configuration_file(
    filename, configuration, include_velocity=True, include_elastics=True, verbose=False
):
    """
    Write a configuration file
    """
    with open(filename, "w") as file:

        file.write(f"[{SIMULATION_BOX}]\n")
        for value in configuration.simulation_box:
            file.write(f"{value}\n")

        file.write(f"[{POSITION}]\n")
        for i in configuration.particles:

            if not (configuration.is_elastic(i) and not include_elastics):
                if POSITION in configuration.particles[i]:
                    pos = configuration.particles[i][POSITION]
                    file.write(f"{pos[0]}\t{pos[1]}\t{pos[2]}\t{pos[3]}\n")
            else:
                if verbose:
                    print(f"Skipping elastic particle {i} in position section.")

        file.write(f"[{VELOCITY}]\n")
        for i in configuration.particles:
            if not (configuration.is_elastic(i) and not include_elastics):
                if include_velocity and VELOCITY in configuration.particles[i]:
                    vel = configuration.particles[i][VELOCITY]
                    file.write(f"{vel[0]}\t{vel[1]}\t{vel[2]}\t{vel[3]}\n")

        file.write(f"[end]\n")



if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python ConfigBuilder.py <config_file> <output_file>")
        sys.exit(1)

    config_file = sys.argv[1]
    out_file = sys.argv[2]
    conf = load_configuration_file(config_file)

    """ """
    write_configuration_file(
        out_file, conf, include_velocity=True, include_elastics=False
    )
    conf2 = load_configuration_file(out_file)
    print("Configuration reloaded from output file: %s"%conf2)
