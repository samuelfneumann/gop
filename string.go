package gop

import (
	"fmt"
	"time"
)

// UnixNano appends an _ followed by the current Unix time in
// nanoseconds to name
func UnixNano(name string) string {
	return fmt.Sprintf("%v_%v", name, time.Now().UnixNano())
}
