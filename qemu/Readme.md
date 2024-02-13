# References

1. install a ubuntu on qemu arm https://wiki.ubuntu.com/ARM64/QEMU
2. https://hackmd.io/@MarconiJiang/qemu_beginner
3. download qemu images https://en.wikibooks.org/wiki/QEMU/Images
4. https://wiki.archlinux.org/title/QEMU

```
sudo qemu-system-aarch64 \
-m 1024 -cpu cortex-a57 -M virt -nographic \
-pflash /usr/share/AAVMF/AAVMF_CODE.fd \
-pflash flash1.img \
-drive if=none,\
file=jammy-server-cloudimg-arm64.img,id=hd0 \
-device virtio-blk-device,drive=hd0 \
-netdev type=tap,id=net0 \
-device virtio-net-device,netdev=net0,mac=$randmac

sudo qemu-system-aarch64 \
-m 1024 -cpu cortex-a57 -M virt -nographic \
file=image_file,format=raw \
,id=hd0 \
-device virtio-blk-device,drive=hd0 \
-netdev type=tap,id=net0 \
-device virtio-net-device,netdev=net0,mac=$randmac
```


5. crate disk for qemu https://serverfault.com/questions/246835/convert-directory-to-qemu-kvm-virtual-disk-image
```
mount -t ext4 -o loop example.img /mnt/example

```

6. https://stackoverflow.com/questions/48989937/qemu-how-to-use-the-virt-board very same idea
```
wget http://archlinuxarm.org/os/ArchLinuxARM-armv7-latest.tar.gz
mkdir archlinux_arm_root
bsdtar -xpf ArchLinuxARM-armv7-latest.tar.gz -C archlinux_arm_root
qemu-img create -f raw disk_img.img 1G
mkfs.ext4 -F disk_img.img
sudo mkdir /mnt/root
sudo mount -o loop disk_img.img /mnt/root
sudo cp -rf archlinux_arm_root/* /mnt/root/
sudo umount /mnt/root
cp archlinux_arm_root/boot/zImage zImage
cp archlinux_arm_root/boot/dtbs/vexpress-v2p-ca9.dtb device_tree.dtb

qemu-system-arm -m 4G -M virt -cpu cortex-a57 \
-kernel zImage \
-dtb device_tree.dtb \
-append "root=/dev/mmcblk0 \
rw roottype=ext4 console=ttyAMA0" \
-drive if=sd,driver=raw,cache=writeback,file=/mnt/arm_qemu_disk -nographic
```

```
qemu-system-aarch64 -M virt -m 4096 -cpu cortex-a57 \
  -kernel installer-linux \
  -initrd installer-initrd.gz \
  -drive if=none,file=/mnt/arm_qemu_disk,id=hd \
  -device virtio-blk-pci,drive=hd \
  -netdev user,id=mynet \
  -device virtio-net-pci,netdev=mynet \
  -nographic -no-reboot
```

```
qemu-system-aarch64 \
    -machine type=virt,virtualization=on \
    -cpu cortex-a57 \
    -kernel Image \
    -initrd initramfs-linux.img \
    -append "rw roottype=ext4 console=ttyAMA0" 
    -m 4096 \
    -smp 4 \
    -accel tcg \
    -device virtio-scsi-pci \
    -drive if=none,driver=raw,cache=writeback,file=/mnt/4c9a5fb6-7d89-4cb8-b727-ab1e9d407286/arm_qemu_disk.img \
    -serial mon:stdio \
    -display none \


    -cpu max,pauth-impdef=on \
    -device scsi-hd,drive=hd \
    -netdev user,id=unet \
    -device virtio-net-pci,netdev=unet \
    -netdev user,id=unet,hostfwd=tcp::2222-:22 \
    -blockdev driver=raw,node-name=hd,file.driver=host_device,file.filename=/mnt/4c9a5fb6-7d89-4cb8-b727-ab1e9d407286/arm_qemu_disk.img \
    -blockdev node-name=rom,driver=file,filename=(pwd)/pc-bios/edk2-aarch64-code.fd,read-only=true \
    -blockdev node-name=efivars,driver=file,filename=$HOME/images/qemu-arm64-efivars
```