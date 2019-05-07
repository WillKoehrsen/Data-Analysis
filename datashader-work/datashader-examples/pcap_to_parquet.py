#!/usr/bin/env python

"""
Convert PCAP output to undirected graph and save in Parquet format.
"""

from __future__ import print_function

import re
import socket
import struct
import sys

import fastparquet as fp
import numpy as np
import pandas as pd


def ip_to_integer(s):
    return struct.unpack("!I", socket.inet_aton(s))[0]


def get_ip_protocol(s):
    if "tcp" in s:
        return "tcp"
    if "UDP" in s:
        return "udp"
    if "EIGRP" in s:
        return "eigrp"
    if "ICMP" in s:
        return "icmp"
    return None


def to_parquet(filename, prefix="maccdc2012"):
    with open(filename) as f:
        traffic = {}
        nodes = set()

        for line in f.readlines():
            if "unreachable" in line:
                continue
            fields = line.split()
            if not fields:
                continue
            if fields[1] != "IP":
                continue
            protocol = get_ip_protocol(line)
            if protocol not in ("tcp", "udp", "eigrp", "icmp"):
                continue
            try:
                addresses = []

                # Extract source IP address and convert to integer
                m = re.match(r'(?P<address>\d+\.\d+\.\d+\.\d+)', fields[2])
                if not m:
                    continue
                addresses.append(ip_to_integer(m.group('address')))

                # Extract target IP address and convert to integer
                m = re.match(r'(?P<address>\d+\.\d+\.\d+\.\d+)', fields[4])
                if not m:
                    continue
                addresses.append(ip_to_integer(m.group('address')))

                nodes = nodes.union(addresses)
                src, dst = sorted(addresses)
                key = (protocol, src, dst)

                # Extract packet size
                nbytes = int(fields[-1])

                if key in traffic:
                    traffic[key] += nbytes
                else:
                    traffic[key] = nbytes
            except:
                pass

        nodes = dict([(node, i) for i, node in enumerate(sorted(nodes))])

        edges = []
        for key in traffic:
            edge = [nodes[key[1]], nodes[key[2]], key[0], traffic[key]]
            edges.append(edge)

        nodes_df = pd.DataFrame(np.arange(len(nodes)), columns=['id'])
        nodes_df = nodes_df.set_index('id')

        edges_df = pd.DataFrame(np.array(edges), columns=['source', 'target', 'protocol', 'weight'])
        edges_df['source'] = pd.to_numeric(edges_df['source'])
        edges_df['target'] = pd.to_numeric(edges_df['target'])
        edges_df['weight'] = pd.to_numeric(edges_df['weight'])
        edges_df['protocol'] = edges_df['protocol'].astype('category')

        fp.write('{}_nodes.parq'.format(prefix), nodes_df)
        fp.write('{}_edges.parq'.format(prefix), edges_df)

if __name__ == '__main__':
    if len(sys.argv) > 2:
        to_parquet(sys.argv[1], prefix=sys.argv[2])
    else:
        to_parquet(sys.argv[1])
