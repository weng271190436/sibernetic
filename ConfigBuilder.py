import pprint

POSITION = "position"
VELOCITY = "velocity"
CONNECTION = "connection"
MEMBRANES = "membranes"
PARTICLE_MEM_INDEX = "particleMemIndex"


def load_configuration_file(filename):
    """
    Load a configuration file
    """
    configuration = {}
    current_section = None
    with open(filename, "r") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "[" in line:
                current_section = line.strip("[]")
                configuration[current_section] = []

            elif current_section == POSITION:
                print("Checking line:", line)
                x, y, z, particle_type = line.split()
                configuration[current_section].append(
                    (float(x), float(y), float(z), float(particle_type))
                )
            else:
                configuration[current_section].append(line)

    pprint.pprint(configuration)
    print(configuration.keys())

    return configuration


def write_configuration_file(
    filename, configuration, include_velocity=True, include_elastics=True
):
    """
    Write a configuration file
    """
    with open(filename, "w") as file:
        for section, items in configuration.items():
            if section == POSITION:
                file.write(f"[{section}]\n")
                for item in items:
                    if int(item[3]) == 2:  # Assuming particle_type 2 is for elastics
                        if not include_elastics:
                            continue
                    else:
                        file.write(f"{item[0]}\t{item[1]}\t{item[2]}\t{item[3]}\n")
            else:
                if not include_velocity and section == VELOCITY:
                    pass
                elif not include_elastics and (
                    section == MEMBRANES
                    or section == CONNECTION
                    or section == PARTICLE_MEM_INDEX
                ):
                    pass
                else:
                    file.write(f"[{section}]\n")
                    for item in items:
                        file.write(f"{item}\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python ConfigBuilder.py <config_file> <output_file>")
        sys.exit(1)

    config_file = sys.argv[1]
    out_file = sys.argv[2]
    conf = load_configuration_file(config_file)

    write_configuration_file(
        out_file, conf, include_velocity=False, include_elastics=False
    )
