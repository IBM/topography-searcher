# for running the landscape search, do something like:
owd=$(pwd)/tmp; mkdir -p tmp/aimnet/ethanol; cd $_
python ../../../run_aimnet_ase.py ethanol

# for performing analysis, do
cd $owd; mkdir analysis; cd $_
python ../../analyse_landscape.py ethanol aimnet2

# to submit and run all landscapes all at once, do something like
for i in aspirin  azobenzene  ethanol  malonaldehyde  paracetamol  salicylic
    do cd $i
        your_submit_script ../../../run_landscapes/run_mace_ase_${i:0:3}* $i
    cd ../
done