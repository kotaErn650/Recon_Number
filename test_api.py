import cv2, numpy as np, base64, json, urllib.request

results = []
for d in range(10):
    img = np.zeros((345, 460, 3), dtype=np.uint8)
    img[:] = 240
    cv2.putText(img, str(d), (190, 230), cv2.FONT_HERSHEY_SIMPLEX, 3, 0, 8, cv2.LINE_AA)
    _, buf = cv2.imencode('.jpg', img)
    b64 = 'data:image/jpeg;base64,' + base64.b64encode(buf).decode()
    data = json.dumps({'image': b64, 'mode': 'camera'}).encode()
    req = urllib.request.Request('http://localhost:5000/api/detect', data=data, headers={'Content-Type': 'application/json'})
    resp = urllib.request.urlopen(req)
    r = json.loads(resp.read())
    results.append(f"{d}->{r.get('digit','?')}({r.get('confidence',0)}%)")

print(' '.join(results))