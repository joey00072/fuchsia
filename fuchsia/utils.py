import socket
import psutil

def get_ip_addresses():
    """Get all network IP addresses with detailed information"""
    ip_info = []
    
    # Get hostname
    hostname = socket.gethostname()
    ip_info.append({
        "interface": "Hostname",
        "ip": hostname,
        "type": "System"
    })
    
    # Get all network interfaces
    for interface, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET:  # IPv4 addresses
                ip_info.append({
                    "interface": interface,
                    "ip": addr.address,
                    "type": "IPv4",
                    "netmask": addr.netmask,
                    "broadcast": addr.broadcast if addr.broadcast else "N/A"
                })
            elif addr.family == socket.AF_INET6:  # IPv6 addresses
                ip_info.append({
                    "interface": interface,
                    "ip": addr.address,
                    "type": "IPv6",
                    "netmask": addr.netmask,
                    "broadcast": "N/A"
                })
    
    return ip_info