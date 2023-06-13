from cmsis_svd.parser import SVDParser


parser = SVDParser.for_packaged_svd('STMicro', 'STM32H743x.svd')  # Establishes a connection to the SVD file 

# Prints all the peripherals and their corresponding addresses in the specific SVD file
for peripheral in parser.get_device().peripherals: 
    print("%s @ 0x%08x" % (peripheral.name, peripheral.base_address))


