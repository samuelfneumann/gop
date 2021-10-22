package gop

import (
	"fmt"
	"time"
)

// Count counts occurrences of strings to append an integer to ensure
// uniqueness in Unique()
var count map[string]uint64

func init() {
	count = make(map[string]uint64)
}

// UnixNano appends an _ followed by the current Unix time in
// nanoseconds to name
func UnixNano(name string) string {
	return fmt.Sprintf("%v_%v", name, time.Now().UnixNano())
}

// UniqueUnixNano is like UnixNano but guarantees that consecutive
// calls using the same string input will always return a unique string.
func UniqueUnixNano(name string) string {
	time.Sleep(time.Nanosecond)
	return UnixNano(name)
}

// Unique appends an _ and an int to name to ensure that it is unique
// from all other strings that have been an argument to Unique().
// If Unique() has never been called with an argument, then the argument
// is returned. If Unique() has already been called with some string
// argument, then Unique() returns the string argument with an "_x"
// appended to the end, where x starts at 1 and increments each time
// Unique() is called with the same argument. For example:
//
//		for i := 0; i < 5; i++ {
//			fmt.Println(Unique("hi"))
//		}
//
// will result in the output
//
//		hi
//		hi_1
//		hi_2
//		hi_3
//		hi_4
func Unique(name string) string {
	suffix, ok := count[name]
	count[name]++

	if ok {
		return fmt.Sprintf("%v_%v", name, suffix)
	}
	return name
}
