package CloudForest

import (
	"sync"
)

/*CatMap is for mapping categorical values to integers.
It contains:

	Map  : a map of ints by the string used for the category
	Back : a slice of strings by the int that represents them

And is embedded by Feature and CatBallotBox.
*/
type CatMap struct {
	privateMap map[string]int //map categories from string to Num
	Back       []string       // map categories from Num to string

	CatMapMut sync.Mutex
}

func NewCatMap() *CatMap {
	return &CatMap{
		privateMap: make(map[string]int, 0),
	}
}

func (cm *CatMap) CopyCatMap() *CatMap {
	cm.CatMapMut.Lock()
	defer cm.CatMapMut.Unlock()

	cp := NewCatMap()
	for k, v := range cm.privateMap {
		cp.privateMap[k] = v
	}
	cp.Back = make([]string, len(cm.Back))
	copy(cp.Back, cm.Back)
	return cp
}

//CatToNum provides the int equivalent of the provided categorical value
//if it already exists or adds it to the map and returns the new value if
//it doesn't.
func (cm *CatMap) CatToNum(value string) (numericv int) {
	cm.CatMapMut.Lock()
	numericv, exsists := cm.privateMap[value]
	if exsists == false {
		numericv = len(cm.Back)
		cm.privateMap[value] = numericv
		cm.Back = append(cm.Back, value)
	}
	cm.CatMapMut.Unlock()
	return
}

//NumToCat returns the catagory label that has been assigned i
func (cm *CatMap) NumToCat(i int) (value string) {
	return cm.Back[i]
}

//NCats returns the number of distinct catagories.
func (cm *CatMap) NCats() (n int) {
	if cm.Back == nil {
		n = 0
	} else {
		n = len(cm.Back)
	}
	return
}
