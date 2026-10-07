// Director's script. Edited live during the roleplay. The page re-reads this file every ~1.5s.
//
// preset:  'showroom' | 'neon' | 'rain' | 'sunset'
// accent:  optional hex for the glow ring and wall strips, e.g. '#ff4fa3'
// label:   holo price tag: { accent, lines: [top, name, bottom] }
// colors:  raw material overrides by name. Materials available on the unit:
//          Hair, Hair_shadow, Hair_highlights, head_skin, hand_skin, Lips, Eyebrows_Primary,
//          Eye_Color_Primary, Eye_Color_Secondary, eye_socket, top_primary, top_secondary, shirt,
//          Top_Buttons, skirt_primary, skirt_secondary, skirt_shadow, Skirt_Buttons, Belt_primary,
//          Belt_Seconadary, Tights, Tights_shadow, Tights_highlight, Shoes, Shoes_highlight,
//          Shoe_Sole, hammer, hammer_1, hammer_2, hammer_3, hammer_5, nails
// pose:    'idle' or { upperarmr: [x, y, z], forearmr: [...], spine: [...], face: [...] } in degrees
// wave:    true to wave with the right hand
// turn:    unit yaw in degrees (0 = facing the camera)
// say:     { id (must change to trigger), speaker, color, text }

export default {
  preset: 'showroom',
  accent: null,
  label: {
    accent: '#00e5ff',
    lines: ['UNIT NB-07 · ANDROID', 'NOBARA KUGISAKI', 'PRICE: ¥2,480,000'],
  },
  // The model ships with black base colors on everything except the hair texture, so the
  // default palette lives here. Override any material by name to restyle the unit.
  colors: {
    Hair: '#ffffff', Hair_shadow: '#5a2a14', Hair_highlights: '#ffd2a0',
    head_skin: '#f1c3a4', head_skin_shadow: '#d99c7e', hand_skin: '#f1c3a4', eye_socket: '#d99c7e',
    Lips: '#b8585f', Eyebrows_Primary: '#5a2a14', Eyebrows_Secondary: '#5a2a14',
    Eye_Color_Primary: '#8a4b22', Eye_Color_Secondary: '#2a1408',
    top_primary: '#2b2320', top_secondary: '#c9b8a6', shirt: '#e8e2d8', Top_Buttons: '#c9a86a',
    skirt_primary: '#1d2233', skirt_secondary: '#2c3350', skirt_shadow: '#11141f', Skirt_Buttons: '#c9a86a',
    Belt_primary: '#3a2a20', Belt_Seconadary: '#c9a86a',
    Tights: '#101014', Tights_shadow: '#0a0a0c', Tights_highlight: '#24242c',
    Shoes: '#4a2f22', Shoes_highlight: '#6a4a36', Shoe_Sole: '#1a1a1a',
    hammer: '#9aa3ad', hammer_1: '#7b848e', hammer_2: '#c0c8d0', hammer_3: '#5b636d', hammer_5: '#3a3f46',
    nails: '#d6dde6',
  },
  pose: 'idle',
  wave: true,
  turn: 0,
  say: {
    id: 2,
    speaker: 'NOBARA KUGISAKI · UNIT NB-07',
    color: '#ff4fa3',
    text: "Oi, you the one with the credits? Took you long enough. I'm Nobara. I hit hard, I talk back, and my warranty doesn't cover hammers to the face. So, you buying me or just staring?",
  },
};
