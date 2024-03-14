$localPath = "`"D:/ETH Zurich/Master Thesis/Surf-GNN/Data/Raw/`""
$remotePath = "/store/sdsc/sd26/LEAP_3D_new/results"
for ($i = 0; $i -le 20; $i++) {
    $fileName = "case_{0:D4}.npz" -f $i
    $remoteFilePath = "$remotePath/$fileName"

    $command = "scp daint:$remoteFilePath $localPath"
    Invoke-Expression $command
}

$command = "scp daint:/store/sdsc/sd26/LEAP_3D_new/3D_data_reader_new.py $localPath"
Invoke-Expression $command

$command = "scp daint:/store/sdsc/sd26/LEAP_3D_new/Params.npy $localPath"
Invoke-Expression $command

$command = "scp daint:/store/sdsc/sd26/LEAP_3D_new/Rough_coord.npz $localPath"
Invoke-Expression $command
