"""
NMEA 2000 (N2K) message decoder - Based on NEAC company implementation
Decodes N2K PGN messages and extracts signals exactly like main.cpp
"""
import pandas as pd
import numpy as np
from typing import Dict, Iterator, List, Tuple


class N2KDecoder:
    """
    Decode NMEA 2000 CAN messages
    Based on NEAC NMEA 2000 CAN Bus Reader implementation (main.cpp)
    """
    
    def __init__(self):
        """Initialize decoder with supported PGN definitions"""
        self.pgn_handlers = self._get_pgn_handlers()
    
    def _get_pgn_handlers(self) -> Dict:
        """
        Get PGN handler mapping - Based on NMEA2000Handlers from main.cpp
        Returns dict of PGN -> handler function name
        """
        return {
            126992: 'SystemTime',
            127245: 'Rudder',
            127250: 'Heading',
            127251: 'RateOfTurn',
            127252: 'Heave',
            127257: 'Attitude',
            127258: 'MagneticVariation',
            127488: 'EngineRapid',
            127489: 'EngineDynamicParameters',
            127493: 'TransmissionParameters',
            127497: 'TripFuelConsumption',
            127501: 'BinaryStatus',
            127505: 'FluidLevel',
            127506: 'DCStatus',
            127508: 'BatteryStatus',
            127513: 'BatteryConfigurationStatus',
            128259: 'Speed',
            128267: 'WaterDepth',
            129025: 'PositionRapid',
            129026: 'COGSOG',
            129029: 'GNSS',
            129033: 'LocalOffset',
            129045: 'UserDatumSettings',
            129283: 'CrossTrackError',
            129284: 'NavigationInfo',
            129285: 'RouteWPInfo',
            129540: 'GNSSSatsInView',
            130306: 'WindSpeed',
            130310: 'OutsideEnvironmental',
            130312: 'Temperature',
            130313: 'Humidity',
            130314: 'Pressure',
            130316: 'TemperatureExt',
            129539: 'GNSSDOPData',
            127237: 'HeadingTrackControl',
            129794: 'AISStaticDataA',
            130577: 'DirectionData',
            130578: 'VesselSpeedComponents',
            129798: 'AISClassAPosition'
        }
    
    # ============================================================================
    # HELPER FUNCTIONS - Byte manipulation
    # ============================================================================
    
    def _bytes_to_uint16(self, data: List[int], offset: int) -> int:
        """Extract unsigned 16-bit integer from bytes"""
        if offset + 1 >= len(data):
            return 0xFFFF
        return data[offset] | (data[offset + 1] << 8)
    
    def _bytes_to_int16(self, data: List[int], offset: int) -> int:
        """Extract signed 16-bit integer from bytes"""
        val = self._bytes_to_uint16(data, offset)
        if val == 0xFFFF:
            return None
        if val > 32767:
            val -= 65536
        return val
    
    def _bytes_to_uint32(self, data: List[int], offset: int) -> int:
        """Extract unsigned 32-bit integer from bytes"""
        if offset + 3 >= len(data):
            return 0xFFFFFFFF
        return data[offset] | (data[offset + 1] << 8) | (data[offset + 2] << 16) | (data[offset + 3] << 24)
    
    def _bytes_to_int32(self, data: List[int], offset: int) -> int:
        """Extract signed 32-bit integer from bytes"""
        val = self._bytes_to_uint32(data, offset)
        if val == 0xFFFFFFFF:
            return None
        # Check for 0x7FFFFFFF (max int32) which is N/A marker for signed int32
        if val == 0x7FFFFFFF:
            return None
        if val > 2147483647:
            val -= 4294967296
        return val
    
    def _check_invalid_uint16(self, val: int) -> bool:
        """Check if uint16 is invalid (0xFFFF)"""
        return val == 0xFFFF
    
    def _check_invalid_uint32(self, val: int) -> bool:
        """Check if uint32 is invalid (0xFFFFFFFF)"""
        return val == 0xFFFFFFFF
    
    # ============================================================================
    # PGN DECODERS - Based on NEAC main.cpp implementation
    # ============================================================================
    
    def _decode_heading(self, data: List[int]) -> Dict:
        """PGN 127250: Vessel Heading - void Heading(const tN2kMsg &N2kMsg)"""
        signals = {}
        # Byte 0: SID
        # Bytes 1-2: Heading (unsigned, 0.0001 radians)
        heading_raw = self._bytes_to_uint16(data, 1)
        if not self._check_invalid_uint16(heading_raw):
            signals['heading'] = (heading_raw * 0.0001) * (180.0 / np.pi)
        else:
            signals['heading'] = np.nan
        
        # Bytes 3-4: Deviation (signed, 0.0001 radians)
        dev_raw = self._bytes_to_int16(data, 3)
        # 0x7FFF (32767) = data not available
        if dev_raw is not None and dev_raw != 32767:
            signals['deviation'] = (dev_raw * 0.0001) * (180.0 / np.pi)
        else:
            signals['deviation'] = np.nan
        
        # Bytes 5-6: Variation (signed, 0.0001 radians)
        var_raw = self._bytes_to_int16(data, 5)
        # 0x7FFF (32767) = data not available
        if var_raw is not None and var_raw != 32767:
            signals['variation'] = (var_raw * 0.0001) * (180.0 / np.pi)
        else:
            signals['variation'] = np.nan
        
        return signals
    
    def _decode_wind_speed(self, data: List[int]) -> Dict:
        """PGN 130306: Wind Speed - void WindSpeed(const tN2kMsg &N2kMsg)"""
        signals = {}
        # Byte 0: SID
        # Bytes 1-2: Wind Speed (unsigned, 0.01 m/s)
        speed_raw = self._bytes_to_uint16(data, 1)
        if not self._check_invalid_uint16(speed_raw):
            signals['wind_speed'] = speed_raw * 0.01  # m/s
        else:
            signals['wind_speed'] = np.nan
        
        # Bytes 3-4: Wind Angle (unsigned, 0.0001 radians)
        angle_raw = self._bytes_to_uint16(data, 3)
        if not self._check_invalid_uint16(angle_raw):
            signals['wind_angle'] = (angle_raw * 0.0001) * (180.0 / np.pi)  # degrees
        else:
            signals['wind_angle'] = np.nan
        
        return signals
    
    def _decode_speed(self, data: List[int]) -> Dict:
        """PGN 128259: Speed - void Speed(const tN2kMsg &N2kMsg)"""
        signals = {}
        # Byte 0: SID
        # Bytes 1-2: Speed, Water Referenced (unsigned, 0.01 m/s)
        sow_raw = self._bytes_to_uint16(data, 1)
        if not self._check_invalid_uint16(sow_raw):
            signals['speed_water'] = sow_raw * 0.01  # m/s
        else:
            signals['speed_water'] = np.nan
        
        # Bytes 3-4: Speed, Ground Referenced (unsigned, 0.01 m/s)
        sog_raw = self._bytes_to_uint16(data, 3)
        if not self._check_invalid_uint16(sog_raw):
            signals['speed_ground'] = sog_raw * 0.01  # m/s
        else:
            signals['speed_ground'] = np.nan
        
        return signals
    
    def _decode_water_depth(self, data: List[int]) -> Dict:
        """PGN 128267: Water Depth - void WaterDepth(const tN2kMsg &N2kMsg)"""
        signals = {}
        # Byte 0: SID
        # Bytes 1-4: Depth (unsigned, 0.01 m)
        depth_raw = self._bytes_to_uint32(data, 1)
        if not self._check_invalid_uint32(depth_raw):
            signals['depth'] = depth_raw * 0.01  # meters
        else:
            signals['depth'] = np.nan
        
        # Bytes 5-6: Offset (signed, 0.001 m)
        offset_raw = self._bytes_to_int16(data, 5)
        if offset_raw is not None:
            signals['offset'] = offset_raw * 0.001  # meters
        else:
            signals['offset'] = np.nan
        
        return signals
    
    def _decode_position_rapid(self, data: List[int]) -> Dict:
        """PGN 129025: Position, Rapid Update - void PositionRapid(const tN2kMsg &N2kMsg)
        
        IMPORTANT: This PGN has NO SID byte!
        Byte 0-3: Latitude (signed, 1e-7 degrees)
        Byte 4-7: Longitude (signed, 1e-7 degrees)
        """
        signals = {}
        # Bytes 0-3: Latitude (signed, 1e-7 degrees)
        lat_raw = self._bytes_to_int32(data, 0)
        if lat_raw is not None:
            signals['latitude'] = lat_raw * 1e-7
        else:
            signals['latitude'] = np.nan
        
        # Bytes 4-7: Longitude (signed, 1e-7 degrees)
        lon_raw = self._bytes_to_int32(data, 4)
        if lon_raw is not None:
            signals['longitude'] = lon_raw * 1e-7
        else:
            signals['longitude'] = np.nan
        
        return signals
    
    def _decode_cog_sog(self, data: List[int]) -> Dict:
        """PGN 129026: COG & SOG - void COGSOG(const tN2kMsg &N2kMsg)"""
        signals = {}
        # Byte 0: SID
        # Byte 1: COG Reference
        # Bytes 2-3: COG (unsigned, 0.0001 radians)
        cog_raw = self._bytes_to_uint16(data, 2)
        if not self._check_invalid_uint16(cog_raw):
            signals['cog'] = (cog_raw * 0.0001) * (180.0 / np.pi)  # degrees
        else:
            signals['cog'] = np.nan
        
        # Bytes 4-5: SOG (unsigned, 0.01 m/s)
        sog_raw = self._bytes_to_uint16(data, 4)
        if not self._check_invalid_uint16(sog_raw):
            signals['sog'] = sog_raw * 0.01  # m/s
        else:
            signals['sog'] = np.nan
        
        return signals
    
    def _decode_attitude(self, data: List[int]) -> Dict:
        """PGN 127257: Attitude - void Attitude(const tN2kMsg &N2kMsg)"""
        signals = {}
        # Byte 0: SID
        # Bytes 1-2: Yaw (signed, 0.0001 radians)
        yaw_raw = self._bytes_to_int16(data, 1)
        if yaw_raw is not None:
            signals['yaw'] = (yaw_raw * 0.0001) * (180.0 / np.pi)
        else:
            signals['yaw'] = np.nan
        
        # Bytes 3-4: Pitch (signed, 0.0001 radians)
        pitch_raw = self._bytes_to_int16(data, 3)
        if pitch_raw is not None:
            signals['pitch'] = (pitch_raw * 0.0001) * (180.0 / np.pi)
        else:
            signals['pitch'] = np.nan
        
        # Bytes 5-6: Roll (signed, 0.0001 radians)
        roll_raw = self._bytes_to_int16(data, 5)
        if roll_raw is not None:
            signals['roll'] = (roll_raw * 0.0001) * (180.0 / np.pi)
        else:
            signals['roll'] = np.nan
        
        return signals
    
    def _decode_rate_of_turn(self, data: List[int]) -> Dict:
        """PGN 127251: Rate of Turn - void RateOfTurn(const tN2kMsg &N2kMsg)"""
        signals = {}
        # Byte 0: SID
        # Bytes 1-4: Rate of turn (signed, 1/32 * 10^-6 rad/s)
        rot_raw = self._bytes_to_int32(data, 1)
        if rot_raw is not None:
            signals['rate_of_turn'] = (rot_raw * 1e-6 / 32.0) * (180.0 / np.pi)  # deg/s
        else:
            signals['rate_of_turn'] = np.nan
        
        return signals
    
    def _decode_rudder(self, data: List[int]) -> Dict:
        """PGN 127245: Rudder - void Rudder(const tN2kMsg &N2kMsg)
        
        IMPORTANT: This PGN has NO SID byte! Structure is:
        Byte 0: Instance
        Byte 1: Direction order (with reserved bits)
        Bytes 2-3: Angle Order (int16, 0.0001 rad)
        Bytes 4-5: Rudder Position (int16, 0.0001 rad)
        
        NMEA 2000 uses 0x7FFF (32767) for "data not available"
        """
        signals = {}
        # Bytes 2-3: Angle Order (signed, 0.0001 radians)
        angle_raw = self._bytes_to_int16(data, 2)
        # 0x7FFF (32767) = data not available
        if angle_raw is not None and angle_raw != 32767:
            signals['rudder_angle_order'] = (angle_raw * 0.0001) * (180.0 / np.pi)
        else:
            signals['rudder_angle_order'] = np.nan
        
        # Bytes 4-5: Rudder Position (signed, 0.0001 radians)
        pos_raw = self._bytes_to_int16(data, 4)
        # 0x7FFF (32767) = data not available
        if pos_raw is not None and pos_raw != 32767:
            signals['rudder_position'] = (pos_raw * 0.0001) * (180.0 / np.pi)
        else:
            signals['rudder_position'] = np.nan
        
        return signals
    
    def _decode_gnss(self, data: List[int]) -> Dict:
        """PGN 129029: GNSS Position Data - void GNSS(const tN2kMsg &N2kMsg)"""
        signals = {}
        # Bytes 7-10: Latitude (signed, 1e-7 degrees)
        if len(data) >= 11:
            lat_raw = self._bytes_to_int32(data, 7)
            if lat_raw is not None:
                signals['gnss_latitude'] = lat_raw * 1e-7
            else:
                signals['gnss_latitude'] = np.nan
        
        # Bytes 11-14: Longitude (signed, 1e-7 degrees)
        if len(data) >= 15:
            lon_raw = self._bytes_to_int32(data, 11)
            if lon_raw is not None:
                signals['gnss_longitude'] = lon_raw * 1e-7
            else:
                signals['gnss_longitude'] = np.nan
        
        # Bytes 15-18: Altitude (signed, 1e-6 meters) - simplified version
        if len(data) >= 19:
            alt_raw = self._bytes_to_int32(data, 15)
            if alt_raw is not None:
                signals['altitude'] = alt_raw * 1e-6
            else:
                signals['altitude'] = np.nan
        
        return signals
    
    # ============================================================================
    # MAIN DECODE FUNCTION
    # ============================================================================
    
    def decode_message(self, can_id: str, data: str) -> Dict[str, float]:
        """
        Decode a single N2K message based on NEAC implementation
        
        Args:
            can_id: CAN ID in hex format (e.g., '0x09fd0219')
            data: Data bytes in hex format (e.g., 'FF A9 00 BA 02 FA FF FF')
        
        Returns:
            Dictionary of signal_name: value
        """
        # Extract PGN from 29-bit CAN ID
        pgn = self._extract_pgn(can_id)
        
        # Parse data bytes
        data_bytes = [int(x, 16) for x in data.split()]
        
        # Decode based on PGN
        signals = {}
        
        if pgn in self.pgn_handlers:
            handler_name = self.pgn_handlers[pgn]
            # Call the appropriate decoder method
            if handler_name == 'Heading':
                signals = self._decode_heading(data_bytes)
            elif handler_name == 'WindSpeed':
                signals = self._decode_wind_speed(data_bytes)
            elif handler_name == 'Speed':
                signals = self._decode_speed(data_bytes)
            elif handler_name == 'WaterDepth':
                signals = self._decode_water_depth(data_bytes)
            elif handler_name == 'PositionRapid':
                signals = self._decode_position_rapid(data_bytes)
            elif handler_name == 'COGSOG':
                signals = self._decode_cog_sog(data_bytes)
            elif handler_name == 'Attitude':
                signals = self._decode_attitude(data_bytes)
            elif handler_name == 'RateOfTurn':
                signals = self._decode_rate_of_turn(data_bytes)
            elif handler_name == 'Rudder':
                signals = self._decode_rudder(data_bytes)
            elif handler_name == 'GNSS':
                signals = self._decode_gnss(data_bytes)
            # Add more handlers as needed
        
        # If no specific decoder, return empty dict
        return signals
    
    def _extract_pgn(self, can_id: str) -> int:
        """
        Extract PGN from 29-bit CAN ID
        
        Args:
            can_id: CAN ID in hex format
        
        Returns:
            PGN number
        """
        # Convert hex string to int
        id_int = int(can_id, 16)
        
        # Extract PGN from bits 8-25 of 29-bit identifier
        pgn = (id_int >> 8) & 0x1FFFF
        pf = (id_int >> 16) & 0xFF
        
        if pf < 240:
            # PDU1 format, PGN uses group extension = 0
            pgn &= 0x1FF00
        
        return pgn
    
    def _iter_complete_messages(self, df: pd.DataFrame) -> Iterator[Tuple[str, str, List[int]]]:
        """
        Iterate over complete N2K payloads, reassembling fast-packet frames when needed.
        
        Args:
            df: Raw CAN dataframe with at least ['Timesteamp', 'ID', 'Data'] columns.
        
        Yields:
            Tuples of (timestamp, can_id, payload_bytes)
        """
        required_columns = {'Timesteamp', 'ID', 'Data'}
        missing_columns = required_columns.difference(df.columns)
        if missing_columns:
            raise ValueError(f"DataFrame missing required columns: {sorted(missing_columns)}")
        
        # Reset index so we can safely look ahead when reassembling fast packets
        frame_df = df.reset_index(drop=True)
        total_frames = len(frame_df)
        idx = 0
        
        while idx < total_frames:
            row = frame_df.iloc[idx]
            raw_data = row['Data']
            
            if not isinstance(raw_data, str):
                idx += 1
                continue
            
            try:
                data_bytes = [int(byte, 16) for byte in raw_data.strip().split()]
            except ValueError:
                # Skip frames with malformed payloads
                idx += 1
                continue
            
            if not data_bytes:
                idx += 1
                continue
            
            timestamp = row['Timesteamp']
            can_id = row['ID']
            dlc = row.get('Length', len(data_bytes))
            
            header = data_bytes[0]
            frame_index = header & 0x1F
            sequence_id = header >> 5
            
            # Attempt fast-packet reconstruction when we detect a frame index 0
            if frame_index == 0 and len(data_bytes) >= 2:
                expected_length = data_bytes[1]
                
                if 0 < expected_length <= 223:
                    payload = data_bytes[2:]
                    consumed = 1
                    
                    # Entire payload fits in first frame
                    if len(payload) >= expected_length:
                        yield timestamp, can_id, payload[:expected_length]
                        idx += consumed
                        continue
                    
                    # Stitch subsequent frames belonging to same fast packet
                    while len(payload) < expected_length and (idx + consumed) < total_frames:
                        next_row = frame_df.iloc[idx + consumed]
                        next_data_raw = next_row['Data']
                        
                        if next_row['ID'] != can_id or not isinstance(next_data_raw, str):
                            break
                        
                        try:
                            next_bytes = [int(byte, 16) for byte in next_data_raw.strip().split()]
                        except ValueError:
                            break
                        
                        if not next_bytes:
                            break
                        
                        next_header = next_bytes[0]
                        next_frame_index = next_header & 0x1F
                        next_sequence_id = next_header >> 5
                        
                        # Fast-packet frames must keep same sequence ID and increasing index
                        if next_sequence_id != sequence_id or next_frame_index != consumed:
                            break
                        
                        payload.extend(next_bytes[1:])
                        consumed += 1
                    
                    if len(payload) >= expected_length:
                        yield timestamp, can_id, payload[:expected_length]
                        idx += consumed
                        continue
                    # If we fall through this point we could not gather the entire payload.
                    # Treat the first frame as a standalone classic message.
            
            # Classic 8-byte (or shorter) frame handling
            yield timestamp, can_id, data_bytes[:dlc]
            idx += 1
    
    def decode_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Decode entire DataFrame of N2K messages
        
        Args:
            df: DataFrame with columns ['Timesteamp', 'ID', 'Data']
        
        Returns:
            DataFrame with decoded signals as columns
        """
        decoded_data = []
        
        for timestamp, can_id, payload in self._iter_complete_messages(df):
            data_str = " ".join(f"{byte:02X}" for byte in payload)
            signals = self.decode_message(can_id, data_str)
            signals['Timestamp'] = timestamp
            decoded_data.append(signals)
        
        result_df = pd.DataFrame(decoded_data)
        
        if not result_df.empty:
            result_df = result_df.reset_index(drop=True)
            result_df = result_df.ffill()
        
        return result_df


if __name__ == "__main__":
    decoder = N2KDecoder()
    print(f"N2KDecoder ready with {len(decoder.pgn_handlers)} PGN definitions")
    print("Supported PGNs:")
    for pgn, handler in sorted(decoder.pgn_handlers.items()):
        print(f"  PGN {pgn}: {handler}")
