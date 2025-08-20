package cse;

public class device {
	private String deviceName;
    private static int deviceCount = 0;

    public device(String name) {
        this.deviceName = name;
        deviceCount++; 
    }

    public String getDeviceName() {
        return deviceName;
    }

    public static int getDeviceCount() {
        return deviceCount;
    }
	



    public static void main(String[] args) {
        device d1 = new device("Smart Bulb");
        device d2 = new device("Smart Thermostat");
        device d3 = new device("Smart Speaker");

        System.out.println("Device 1: " + d1.getDeviceName());
        System.out.println("Device 2: " + d2.getDeviceName());
        System.out.println("Device 3: " + d3.getDeviceName());

        System.out.println("Total devices connected: " + device.getDeviceCount());
    }
}